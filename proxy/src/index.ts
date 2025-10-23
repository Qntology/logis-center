import { Node, parseHTML } from 'linkedom'

import { gzip, ungzip } from 'pako'

import { ethers } from 'ethers'


/*
	--- 결제 타입 ---
		$user
		$team

		+++ 결제 플로우 만들어야함

	***selector 가 같은데 계속 풀 html 문서 전송막기

	사용자가 안사용하는 벡터 DB 자동 정리하는 기능 추가하기


*/

/*
	무료 oauth 토큰

	유료 logis 토큰

	아이콘 설명
		✨ 로지스 센터 확장프로그램 & AI
		🖥️ 데스크탑
		📱 휴대폰


	어플리케이션
		활성화 주소에서만 ✨ 버튼 자동화 (기존 쇼핑몰 관리자에서 사용 가능하게)

		오른쪽 하단의 ✨ 클릭시 동기화 시작
		
		네비게이션
			✨ 로그인
			쇼핑몰 추가
				추가완료시 쇼핑몰 파비콘과 쇼핑몰 이름 표시
				🛍️ 쇼핑몰 주문목록 링크 추가
				📦 쇼핑몰 재고목록 링크 추가
				🖥️ 휴대폰 동기화 QR 버튼(보안 경고창 띄우고 QR 노출하기)
				📱 송장 조회
				📱 재고 조회 & 재고 증감

		이벤트
			🖥️ 송장 출력
			✨ 주문 조회
			✨ 재고 조회

			재고 타입
				- 자체 등록
				- AI 등록

	휴대폰
		송장 스캔(오프라인)
			발주, 발송 사용자가 선택
				- 송장번호는 단한번만 추가되며 추가, 삭제만 가능
				- 발주시 재고 추가됨
				- 발송시 재고 차감
				- draft 저장소에 등록
					type draft 값으로 등록

		AI 보정
			* 보정은 처음 혹은 정상작동하지 않으면 동작합니다.
			* 요청은 html 문서를 서버에 전송하면 json 구조로 리턴합니다.

			오프라인
				송장 스캔
					- draft 저장소에 등록
						type draft 값으로 등록



	거의 수동
	- 크롬 익스텐션에서
		발주시 송장번호를 타입(상품번호, 주문번호)에 마킹
			예시 여러 상품 조합인 경우 여러개 등록하는 형식이여야함

			배송상태는 실제 송장을 스캔하면 완료로 체크


	일차적으로 
		쇼핑몰 주문 관리 페이지


	OCR 시 
		DRAFT로 등록하고, 재고 여부 확인후 병합 


	1000회 limit 요청 차게 될수도 있으니 fetch 요청하는것으로 우회하기


		


	team.data.base.graph
		page.ref = 레퍼러
		page.ref 간에 연결을 프로세스로 보여줌


		오른쪽 
			UI 채팅으로 노출시킬 내용은 스캔, 프롬프트 결과만

		왼쪽영역
			노출할 내용
				- 페이지 테이블
				- 연관 테이블
				- 메모

		실제 page.ref 값
			shopping_mall.host
				> goods
					// tracking 테이블에서 해당 list, detail id 값 참조해서 날짜 값을 기준으로 평가함

					> list 플로우
						?type=order &created_at = ${Today} &limit = 100
						
						~~ 상품명	최근 24시간 주문	재고	상태
						~~ 상품 A	23건 (+15%)	12	🔥 판매호조
						~~ 상품 B	0건 (-100%)	150	⚠️ 판매정체
						
						> detail
							~~ “최근 24시간 주문 15건”, “이번 주 82건, 지난주 대비 +12%”
							~~ 조회 트래킹을 해보세요!

				> order
					> list 플로우
						?type=tracking &created_at = ${Today} &limit = 100
						
						~~ 주문 대기 상태 상품 리스트 표시
						~~ 작업자 상태
						++ 작업 프로세스 플로우 메모
						++ tracking draft 노출

						> detail 플로우
							0. event 있으면 플로우 표시
							1. goods 표시
							2. order 표시
							3. tracking 표시

							~~ 주문 대기 상태 상품 정보 표시
							~~ 작업 상태 or 설정

							++ 작업 프로세스 플로우 메모

							++ tracking draft 노출

*/


function crc32(s) { var polynomial = arguments.length < 2 ? 0x04C11DB7 : arguments[1], initialValue = arguments.length < 3 ? 0xFFFFFFFF : arguments[2], finalXORValue = arguments.length < 4 ? 0xFFFFFFFF : arguments[3], crc = initialValue, table = [], i, j, c; function reverse(x, n) { var b = 0; while (n) { b = b * 2 + x % 2; x /= 2; x -= x % 1; n--; } return b; } for (i = 256; i >= 0; i--) { c = reverse(i, 32); for (j = 0; j < 8; j++) { c = ((c * 2) ^ (((c >>> 31) % 2) * polynomial)) >>> 0; } table[i] = reverse(c, 32); } for (i = 0; i < s.length; i++) { c = s.charCodeAt(i); if (c > 255) { throw new RangeError(); } j = (crc % 256) ^ c; crc = ((crc / 256) ^ table[j]) >>> 0; } return (crc ^ finalXORValue) >>> 0; }


const randomKey = function(){
	var key = Math.random().toString()

	return parseInt(key.replace("0.",""))
}

/**
 * 두 객체를 병합하여 새로운 객체를 반환합니다.
 * obj2에 유효한 값이 있다면 obj1의 값에 관계없이 무조건 덮어씁니다.
 * obj2의 값이 비어있다면(null, undefined, '') obj1의 값을 유지합니다.
 *
 * @param {Object} obj1 기본 객체
 * @param {Object} obj2 덮어쓸 값(소스)을 가진 객체
 * @returns {Object} 병합된 새로운 객체
 */
function mergeNode(obj1, obj2) {
	// '비어있다'는 기준은 null, undefined, 빈 문자열('')로 정의합니다.
	const isEmpty = (value) => value === null || value === undefined || value === '';

	// 1. obj1의 모든 속성을 복사하여 새로운 객체를 생성합니다.
	const merged = { ...obj1 };

	// 2. obj2의 모든 키를 순회하며 병합 작업을 수행합니다.
	for (const key in obj2) {
		if (obj2.hasOwnProperty(key)) {
			const value2 = obj2[key];

			// **핵심 로직**
			// obj2의 값(value2)이 비어있지 않다면 (유효하다면)
			if (!isEmpty(value2)) {
				// obj1의 값의 유효성과 관계없이 무조건 obj2의 값으로 덮어씁니다.
				merged[key] = value2;
			}
			// obj2의 값이 비어있다면 아무 작업도 하지 않아
			// 기존 merged 객체(obj1에서 복사됨)의 값이 유지됩니다.
		}
	}

	return merged;
}

const image2json = function(type, address){
	if(type == "tracking"){
		return `convert the shipping label image to fit the dataset JSON structure. Return only the JSON structure result, no explanation.{
			no:Tracking Number(운송장 번호 or 运单号 or 運單號 or 伝票番号 or Número de seguimiento or Numéro de suivi or Sendungsnummer or Номер накладной or Número de rastreamento or Numero di tracciamento or رقم التتبع or Số vận đơn or Nomor resi or หมายเลขติดตามพัสดุ) | string,
			recipient_address : ${JSON.stringify(address)},
			recipient_match : shipping label recipient address match. Ruled the same despite different floor levels | true/false,
			text : summarize including the shipping label contents. Filter the addresses included in the summary information to District-level and up | string,
			barcode : [barcode number | string]
		}`
	}
}


/*
- Segment the natural language content into core types of nested connected context based on the type table schema and extract from those segmented contexts to fit the dataset JSON structure based on declared types. no explanation.

convert the natural language content to fit the dataset JSON structure. no explanation.
{ 
	context : [
		{
			language : "korean",
			type:'sales' or 'order' or 'goods' or 'tracking' or 'view' or 'review' or 'coupon' or 'event' or '',
			text:Segment the natural language content into single-type contexts
		},...
	]
}
올해 여름 이벤트로 판매된 제품중에서 무거운 제품으로 5000원 이하로 많이 팔린 제품 중에서 리뷰를 남긴 고객의 메세지도 보여줘
*/

const para2graph = function(language){
	return `convert the natural language content to fit the dataset JSON structure. no explanation.
	{ 
		context : [
			{
				language : "${language}",
				type:'sales' or 'order' or 'goods' or 'tracking' or 'view' or 'review' or 'coupon' or 'event' or '',
				text:Segment the natural language content into single-type contexts
			},...
		]
	}`
}

/*
convert the natural language content to fit the dataset JSON structure. no explanation.
# date filter : The date value is set by referencing both the natural language's implied time period and the region value against the current time (2025−09−25T18:23:46.364Z); it will be marked as null if a value is absent
# status : 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete'
# substantial : 'size' or 'weight' or 'shipping_fee' or 'shipping_duration' or 'sale_price' or 'supply_price' or 'low_stock_threshold' or 'discount' or 'min_order_amount' or 'max_discount_amount' or 'usage_limit' or 'usage_per' or ''
# find : 'many' or 'few' or 'much' or 'little' or 'heavy' or 'light' or ''
{
	"context": [
		{
			"region": "korean",
			"language": "korean",
			"type": "event",
			"text": "올해 여름 이벤트",
			"status":null,
			"substantial":null,
			"find":null,
			"condition" : {
				"date":{
					"eq":yyyy-MM-dd'T'HH:mm:ss,"lte":yyyy-MM-dd'T'HH:mm:ss,"gte":yyyy-MM-dd'T'HH:mm:ss
				},
				"quantity":{
					"eq":0,"lte":0,"gte":0
				},
				"price":{
					"currency":'',
					"eq":0,"lte":0,"gte":0
				}
			}
		},
		{
			"region": "korean",
			"language": "korean",
			"type": "sales",
			"text": "판매된 제품중에서",
			"status":null,
			"substantial":null,
			"find":null,
			"condition" : {
				"date":{
					"eq":yyyy-MM-dd'T'HH:mm:ss,"lte":yyyy-MM-dd'T'HH:mm:ss,"gte":yyyy-MM-dd'T'HH:mm:ss
				},
				"quantity":{
					"eq":0,"lte":0,"gte":0
				},
				"price":{
					"currency":'',
					"eq":0,"lte":0,"gte":0
				}
			}
		},
		{
			"region": "korean",
			"language": "korean",
			"type": "goods",
			"text": "무거운 제품으로 5000원 이하로 많이 팔린 제품",
			"status":null,
			"substantial":null,
			"find":null,
			"condition" : {
				"date":{
					"eq":yyyy-MM-dd'T'HH:mm:ss,"lte":yyyy-MM-dd'T'HH:mm:ss,"gte":yyyy-MM-dd'T'HH:mm:ss
				},
				"quantity":{
					"eq":0,"lte":0,"gte":0
				},
				"price":{
					"currency":'',
					"eq":0,"lte":0,"gte":0
				}
			}
		},
		{
			"region": "korean",
			"language": "korean",
			"type": "review",
			"text": "리뷰를 남긴 고객의 메세지도 보여줘",
			"status":null,
			"substantial":null,
			"find":null,
			"condition" : {
				"date":{
					"eq":yyyy-MM-dd'T'HH:mm:ss,"lte":yyyy-MM-dd'T'HH:mm:ss,"gte":yyyy-MM-dd'T'HH:mm:ss
				},
				"quantity":{
					"eq":0,"lte":0,"gte":0
				},
				"price":{
					"currency":'',
					"eq":0,"lte":0,"gte":0
				}
			}
		}
	]
}


올해 여름 이벤트로 판매된 제품중에서 무거운 제품으로 5000원 이하로 많이 팔린 제품 중에서 리뷰를 남긴 고객의 메세지도 보여줘

'여름 시즌' 기획전에 포함된 상품들 중, 상세 페이지 조회수는 상위 20%에 속하지만 구매 전환율이 1% 미만인 상품들만 따로 보여줘. 원인 분석이 시급해
*/

const graph2contexts = function(current){
	return `convert the natural language content to fit the dataset JSON structure. no explanation.
	# date filter : The date value is set by referencing both the natural language's implied time period and the region value against the current time (${current}); it will be marked as null if a value is absent
	# status : 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error'
	# substantial : 'size' or 'weight' or 'shipping_fee' or 'shipping_duration' or 'sale_price' or 'supply_price' or 'low_stock_threshold' or 'discount' or 'min_order_amount' or 'max_discount_amount' or 'usage_limit' or 'usage_per' or ''
	# find : 'many' or 'few' or 'much' or 'little' or 'heavy' or 'light' or ''`
}


const list2json = function(language){
	return `
		type:'order' or 'goods' or 'tracking' or 'search' or 'review' or 'member' or 'coupon' or 'event' or '',
		isDetail:is detail page | true/false,
		node:Item parent list CSS selector excluding ads,
		item:Item CSS selector excluding ads,
		more:Item detail link CSS selector,
		next:items next button CSS selector,
		text:Summarize the contents of the items array in ${language},
		items: [
			if (type is 'tracking' or 'review' or 'member') {
				status:'start' or 'progress' or 'stop' or 'cancel' or 'return',
				id:Refer to the ID value from the link or an attribute | string,
				title:author and content | string, 
				link:URL includes manage path additional link | string,
				date:yyyy-MM-dd'T'HH:mm:ss | string,
			}
			if (type is 'order' or 'goods') {
				status:'show' or 'progress' or 'remove' or 'hide' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
				link:URL includes manage path additional link | string,
				id:Refer to the ID value from the link or an attribute | string,
				title:title | string, 
				sale_price:sale price | number,
				supply_price:supply price | number,
				currency:ISO 4217 Currency Code | string,
				quantity:item stock quantity | number,
				tracking_number:Tracking Number(운송장 번호 or 运单号 or 運單號 or 伝票番号 or Número de seguimiento or Numéro de suivi or Sendungsnummer or Номер накладной or Número de rastreamento or Numero di tracciamento or رقم التتبع or Số vận đơn or Nomor resi or หมายเลขติดตามพัสดุ) | string,
				date:yyyy-MM-dd'T'HH:mm:ss | string,
			}
			if (type is 'coupon' or 'event') {
				status : 'show' or 'progress' or 'hide' or 'stop' or 'cancel' or 'expire' or 'complete' or 'error',
				id:Refer to the ID value from the link or an attribute | string,
				title:type based item title, 
				started_at:yyyy-MM-dd'T'HH:mm:ss,
				expired_at:yyyy-MM-dd'T'HH:mm:ss,
			}
		] 
	`
}


const item2json = function(type){
	if(type == 'tracking'){
		return ` 
			node:detail page element CSS selector,
			status:'draft' or 'progress' or 'return' or 'complete' or 'error',
			id:tracking number | string,
			title:${type} goods title | string, 
			sender_name:sender_name | string,
			sender_address:sender_address | string,
			sender_phone:sender_phone | string,
			recipient_name:recipient_name | string,
			recipient_address:recipient_address | string,
			recipient_phone:recipient_phone | string,
			package_width:Package width | number,
			package_height:Package height | number,
			package_length:Package length | number,
			package_weight:Package weight | number,
			carrier:carrier name translated into English | string,
			shipping_fee:Shipping cost | number,
			shipping_method:'standard' or 'express' or 'same_day' or 'pick_up' or 'freight,
			shipping_duration:Estimated delivery days | number,
			bundle_shipping:Allow combined shipping | string,
			shipping_date:yyyy-MM-dd'T'HH:mm:ss | string,
		`
	}else if(type == 'goods'){
		return `
			node:detail page element CSS selector,
			id:Refer to the ID value from the link or an attribute or input value | string,
			status:'draft' or 'show' or 'hide' or 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
			payment_method:payment method | string,
			bank:bank company name or '' | string,
			card:card company name or '' | string,
			code:product constant code | string,
			model_name:product Model name | string,
			brand_name:product Brand name | string,
			condition:['new' or 'used' or 'lease' or 'rental' or 'refurbish'],
			description:product Full description (HTML allowed) | string,
			short_description:product short description | string,
			tags:[{ tag : product keyword or tag | string }],
			origin_country:product Country of origin/manufacture | string,
			manufacturer:product Manufacturer name | string,
			release_date:Product release date(yyyy-MM-dd'T'HH:mm:ss) | string,
			manufacture_date:product Date(yyyy-MM-dd'T'HH:mm:ss) of manufacture | string,
			expiration_date:product Expiration or use-by date(yyyy-MM-dd'T'HH:mm:ss) | string,
			gtin:product Global Trade Item Number | string,
			mpn:product Manufacturer Part Number | string,
			barcode:product Barcode value | string,
			sale_price:product sale price | number,
			supply_price:product supply price | number,
			currency:ISO 4217 Currency Code | string,
			compare_at_price:product Original price for showing discounts | number,
			quantity:product Inventory quantity | number,
			stock_keeping_unit: Stock Keeping Unit | string,
			low_stock_threshold:product Low stock alert threshold | number,
			unit:product Selling unit | string,
			tax_included:product Whether tax | number,
			tax_code:product Tax code for region-specific rules | string,
			main_image_url:Main product image URL | string,
			additional_image_url:additional product image URL | string,
			video_url:product Promotional video URL | string,
			carrier:product carrier name translated into English | string,
			shipping_fee:product Shipping cost | number,
			shipping_method:'standard' or 'express' or 'same_day' or 'pick_up' or 'freight,
			shipping_duration:product Estimated delivery days | number,
			bundle_shipping:product Allow combined shipping | string,
			product_width:Package width(cm) | number,
			product_height:Package height(cm) | number,
			product_length:Package length(cm) | number,
			product_weight:Package weight(kg) | number,
			options:[
				{
					name : option name | string,
					inputs:[{
						input:option input value | string,
					}]
				}
			],
			additional_goods:[
				{
					link:URL includes manage path additional product link | string
				}
			],
			title:product based title | string,
			date:yyyy-MM-dd'T'HH:mm:ss | string,
		`
	}else if(type == 'order'){
		return `
			node:detail page element CSS selector,
			id:Refer to the ID value from the link or an attribute or input value | string,
			tracking_number:tracking number | string,
			status:'draft' or 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
			goods:[{
				title:goods title | string,
				link:URL includes manage path additional goods link | string,
				id:Refer to the ID value from the link or an attribute | string,
			}],
			sender_name:sender_name | string,
			sender_address:sender_address | string,
			sender_phone:sender_phone | string,
			recipient_name:recipient_name | string,
			recipient_address:recipient_address | string,
			recipient_phone:recipient_phone | string,
			bank:bank company name | string,
			card:card company name | string,
			order_date:order date | string,
			payment_date:payment date or '' | string,
			payment_method:'C.O.D.' or 'CARD' or 'BANK' or '' | string,
			payment_origin:Payment Gateway Service Name or '' | string,
			date:yyyy-MM-dd'T'HH:mm:ss | string
		`
	}else if(type == 'coupon' || type == 'event'){
		return `
			node:detail page element CSS selector,
			id:Refer to the ID value from the link or an attribute or input value | string,
			type:'percentage' or 'fixed_amount' or 'free_shipping' or '',
			status:'draft' or 'progress' or 'stop' or 'cancel' or 'expire' or 'complete' or 'error',
			title:${type} item title | string, 
			started_at:yyyy-MM-dd'T'HH:mm:ss | string,
			expired_at:yyyy-MM-dd'T'HH:mm:ss | string,
			code:${type} code used at checkout | string,
			discount:Discount value | number,
			quantity:${type} quantity | number
			usage_limit:Total usage limit for the coupon | number,
			usage_per:Usage limit per customer | number,
			new_customer_only:new customer only | boolean
			min_order_amount:Minimum order amount required to apply coupon | number,
			max_discount_amount:Maximum discount limit allowed for the coupon | number,
			region_restrictions:region restrictions | boolean
		`
	}else if(type == 'review' || type == 'member'){
		return `
			node:detail page element CSS selector,
			id:Refer to the ID value from the link or an attribute or input value | string,
			status:'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
			name:${type} name | string,
			title:${type} item title | string, 
			completed:order complete | boolean,
			created_at:yyyy-MM-dd'T'HH:mm:ss
		`
	}
}



const form2json = function(type){
	if(type == 'tracking' || type == 'review' || type == 'member'){
		return `
			node:${type} item parent list CSS selector excluding ads,
			item:${type} item CSS selector excluding ads,
			title:${type} item title CSS selector excluding ads, 
			date:${type} item date value CSS selector
		`
	}else if(type == 'goods'){
		return `
			display:product display status CSS selector,
			code:product constant code CSS selector,
			model_name:Model name CSS selector,
			brand_name:Brand name CSS selector,
			usedType:usedType CSS selector,
			description:Full description (HTML allowed) CSS selector,
			short_description : short description CSS selector,
			tags:tag or keyword CSS selector,
			origin_country:Country of origin/manufacture CSS selector,
			manufacturer:Manufacturer name CSS selector,
			release_date:Product release date CSS selector,
			manufacture_date:Date of manufacture CSS selector,
			expiration_date:Expiration or use-by date CSS selector,
			gtin:Global Trade Item Number CSS selector,
			mpn:Manufacturer Part Number CSS selector,
			barcode:Barcode value CSS selector,
			sale_price:sale price CSS selector,
			supply_price:supply price CSS selector,
			compare_at_price:Original price for showing discounts CSS selector,
			quantity:Inventory quantity CSS selector,
			stock_keeping_unit:Stock Keeping Unit CSS selector,
			low_stock_threshold:Low stock alert threshold CSS selector,
			unit:Selling unit CSS selector,
			tax_included:Whether tax CSS selector,
			tax_code:Tax code for region-specific rules CSS selector,
			main_image_url:Main product image URL CSS selector,
			additional_image_url:additional product image URL CSS selector,
			video_url:Promotional video URL CSS selector,
			carrier:carrier CSS selector,
			shipping_fee:Shipping cost CSS selector,
			shipping_method:Shipping method CSS selector,
			shipping_duration:Estimated delivery days CSS selector,
			bundle_shipping:Allow combined shipping CSS selector,
			product_width:product width CSS selector,
			product_height:product height CSS selector,
			product_length:product length CSS selector,
			product_weight:product weight CSS selector,
			fulfillment_service:Fulfillment provider CSS selector,
			options:[{
				name : option name CSS selector,
				inputs:[{
					input:option input CSS selector,
				}]
			}],
			additional_goods:[{
				link:URL includes manage path additional goods link CSS selector
			}],
			title:goods title CSS selector,
			date:goods date(yyyy-MM-dd'T'HH:mm:ss) CSS selector
		`
	}else if(type == 'order'){
		return `
			status:${type} status CSS selector,
			order_products:[{
				title:product title CSS selector,
				options:[{
					name : product option name CSS selector,
					option:product option value CSS selector,
				}],
				link:URL includes manage path additional product link CSS selector
			}],
			date:order date CSS selector
		`
	}else if(type == 'coupon' || type == 'event'){
		return `
			status:${type} status CSS selector,
			title:${type} item title CSS selector, 
			start_at:${type} item start date value(yyyy-MM-dd'T'HH:mm:ss) CSS selector,
			end_at:${type} item end date value(yyyy-MM-dd'T'HH:mm:ss) CSS selector,
			type:Type of discount CSS selector,
			code:${type} code used at checkout CSS selector,
			discount:Discount value input CSS selector,
			new_customer_only:new customer only input CSS selector
			min_order_amount:Minimum order amount required to apply coupon value input CSS selector,
			max_discount_amount:Maximum discount limit allowed for the coupon value input CSS selector,
			usage_limit:Total usage limit for the coupon value input CSS selector,
			usage_per:Usage limit per customer value input CSS selector
			region_restrictions:region restrictions value input CSS selector
		`
	}
}

const context2results = function(context, results, language){
	var condition = ''

	if(obj.condition){
		condition = `condition : ${JSON.stringify(context.condition)}`
	}

	return `{
		search : {
			text : '${context.text}',
			query : {
				${condition}
			},
			results : ${JSON.stringify(results)}
		}
	}
	return JSON Structure {
		results : [find the content corresponding to {search.text} in {search.results}],
		text : Please summarize the search results and the context in ${language}.
	}
	`
}


const semantic_prompt_system = function(language){
	return `Converts and returns the JSON structure as natural language in ${language}. no explanation.`
}



const isDiff = (obj1, obj2) => {
	// If both objects are null or undefined, they are not considered different.
	if (!obj1 && !obj2) {
		return false;
	}

	// If one is falsy and the other isn't, they are different.
	if (!obj1 || !obj2) {
		return true;
	}

	const keys1 = Object.keys(obj1);
	const keys2 = Object.keys(obj2);

	// If the number of keys is different, the objects are different.
	if (keys1.length !== keys2.length) {
		return true;
	}

	// Iterate over keys to check for differences.
	for (const key of keys1) {
		// Check for specific buffer comparison for keys named 'data'.
		if (key === 'data' && Buffer.isBuffer(obj1[key]) && Buffer.isBuffer(obj2[key])) {
			// Use Buffer.equals() for efficient byte-by-byte comparison.
			if (!obj1[key].equals(obj2[key])) {
				return true;
			}
		} else if (typeof obj1[key] === 'object' && typeof obj2[key] === 'object') {
			// Recursively call isDiff for nested objects.
			if (isDiff(obj1[key], obj2[key])) {
				return true;
			}
		} else if (obj1[key] !== obj2[key]) {
			// If values are not equal, the objects are different.
			return true;
		}
	}

	// If no differences are found, the objects are the same.
	return false;
};


// const getCommonAncestor = (elements) => {
//  if (!elements || elements.length === 0) {
//      return null;
//  }

//  // Start with the first element's parent as the potential common ancestor
//  let ancestor = elements[0].parentNode;

//  // Loop through all elements
//  for (let i = 1; i < elements.length; i++) {
//      // Check if the current ancestor contains the next element
//      // If not, move up the tree from the first element
//      if (!ancestor.contains(elements[i])) {
//          ancestor = ancestor.parentNode;
//          // Restart the loop to re-check all elements with the new ancestor
//          i = 0; 
//      }
//  }

//  return ancestor;
// };

// // Example usage:
// const elements = document.querySelectorAll('.item');
// const commonAncestorElement = getCommonAncestor(elements);

/*
	상태
		대기
			"draft"

		삭제
			"delete"


*/

/*
	reviews 정보는 title에 작성자 이름과  리뷰 내용 합쳐서 넣기


	vector 메타데이터로 저장해야할 분류
		'review' or 'member' or 'coupon' or 'event'

		
		amount 0

*/ 



var extractNumbersRegex = /\d+/g;

function getZeroUTC(date, day) {
	date.setDate(date.getDate() - day)

	date.setUTCHours(0)
	date.setUTCMinutes(0)
	date.setUTCSeconds(0)
	date.setUTCMilliseconds(0)

	return date.getTime() // 'YYYY-MM-DDTHH:mm:ss.sssZ'
}


function hashId(text){
	if(typeof text == "undefined"){
		var account = ethers.Wallet.createRandom()
		text = account.privateKey
	}

	var hashMessage = ethers.hashMessage(text)

	return ethers.computeAddress(hashMessage).toLowerCase()
}

function hasUndefinedValue(obj){
	for (const key in obj) {
		if (obj.hasOwnProperty(key) && obj[key] === undefined) {
			return true;
		}
	}
	return false;
}

function containsUnsupportedChars(str) {
	// 정규 표현식: /[^a-zA-Z0-9\s!@#$%^&*()_+\-=\[\]{}|;:'",.<>/?~`]/
	// ^: 대괄호 안에서 '부정(not)'을 의미합니다.
	// a-zA-Z: 영어 알파벳
	// 0-9: 숫자
	// \s: 공백, 탭 등 모든 공백 문자
	// !@#$%^&*()_+\-=\[\]{}|;:'",.<>/?~`: 허용할 특수문자 목록입니다.
	// 정규식에서 특별한 의미를 갖는 문자(-, [, ], \, ^)는 앞에 역슬래시(\)를 붙여 이스케이프 처리해야 합니다.
	const regex = /[^a-zA-Z0-9\s!@#$%^&*()_+\-=\[\]{}|;:'",.<>/?~`]/;
	
	// test() 메서드는 문자열에서 정규식과 일치하는 부분이 있으면 true, 없으면 false를 반환합니다.
	return regex.test(str);
}



function parseCondition(obj, col, condition){
	var val = function(k, v){
		if(k == "date"){
			return new Date(v).getTime()
		}else{
			return v
		}
	}

	if(obj.gte && obj.lte){
		condition += ` "${col}" >= ${val(col,obj.gte)} AND ${col} <= ${val(col,obj.lte)}`
	}else if(obj.gte){
		condition += `"${col}" >= ${val(col,obj.gte)}`
	}else if(obj.lte){
		condition += `"${col}" <= ${val(col,obj.lte)}`
	}else if(obj.eq){
		condition += `"${col}" = ${val(col,obj.eq)}`
	}

	return condition;
}


async function Sleep(ms) {
	return new Promise(resolve => setTimeout(resolve, ms))
}



function $Contains(text, selector){
	var arr = []

	var $target = selector ? document.querySelector(selector) : document.body;

	$target.querySelectorAll("*").forEach((el) => {
		let str = el.innerText;
		
		if(str?.includes(text)) {
			//item.classList.add("on");
			arr.push(el);
		} 
	})

	return arr
}

/**
 * HTML을 간결한 Pug 코드로 변환하는 최종 함수
 * @param {string} body - 변환할 HTML 문자열
 * @returns {string} 허용된 속성(id, class, img src, a href) 외에는 제거되고, 불필요한 태그가 정리된 Pug 코드
 */
function convertHtmlToCleanPug(body) {
	try {
		var { document } = parseHTML(`<html><body>${body}</body></html>`);

		const pugLines = generatePugLines(document.body.childNodes, 0);

		return pugLines.join('\n');

	} catch (error) {
		console.error('변환 중 오류가 발생했습니다:', error);
		return '';
	}
}


/**
 * DOM 노드를 재귀적으로 순회하며 Pug 라인 배열을 생성하는 내부 함수
 * @param {NodeListOf<ChildNode>} nodes - 변환할 DOM 노드 리스트
 * @param {number} indentLevel - 들여쓰기 레벨
 * @returns {string[]} 생성된 Pug 라인 배열
 */
function generatePugLines(nodes, indentLevel) {
	// 들여쓰기 문자 (공백 4칸)
	const indent = '    '.repeat(indentLevel); 
	let lines = [];

	nodes.forEach(node => {
		// 1. Element 노드 처리
		if (node.nodeType === Node.ELEMENT_NODE) {
			const tagName = node.tagName.toLowerCase();

			// --- base64 이미지를 포함하는 img 태그 제외 ---
			const src = node.getAttribute('src');
			if (tagName === 'img' && src && src.includes('base64')) {
				return; // src에 'base64'가 포함된 img 태그는 변환에서 건너뜁니다.
			}

			// 불필요한 태그들을 만나면 건너뛰기
			// input, textarea는 이제 포함됩니다.
			if (['script', 'style', 'link', 'noscript', 'iframe', 'button'].includes(tagName)) {
				return;
			}

			// --- 허용된 속성만 Pug 문법으로 변환 ---
			let attributesString = '';
			const otherAttributes = [];

			// ID 속성 처리 (#my-id)
			if (node.id) {
				attributesString += `#${node.id}`;
			}

			// Class 속성 처리 (.class1.class2)
			if (node.classList.length > 0) {
				attributesString += `.${Array.from(node.classList).join('.')}`;
			}

			// NamedNodeMap을 Array로 변환하여 모든 속성을 순회합니다.
			Array.from(node.attributes).forEach(attr => {
				const attrName = attr.name;
				const attrValue = attr.value;

				// 기본적으로 포함할 속성들: input, a, img, textarea의 주요 속성 포함
				const alwaysInclude = [
					'src', 'href', 'type', 'name', 'value', 'placeholder', 
					'checked', 'selected', 'disabled', 'readonly', 'rows', 'cols'
				];

				// ID와 Class는 이미 처리되었으므로 제외
				if (attrName === 'id' || attrName === 'class') {
					return;
				}

				if (attrName.startsWith('data-') || alwaysInclude.includes(attrName)) {
					// Boolean 속성 처리 (ex: disabled, checked, readonly)
					if (['checked', 'selected', 'disabled', 'readonly'].includes(attrName) && (attrValue === '' || attrValue === attrName)) {
						otherAttributes.push(`${attrName}`); // 값 없이 속성 이름만 추가 (Pug의 Boolean 속성 표기)
					} else if (attrValue) { // 속성값이 비어있지 않은 경우에만 추가
						// 따옴표 안에 따옴표가 있는 경우 이스케이프 필요 (여기서는 단순하게 큰따옴표로 처리)
						const safeValue = attrValue.replace(/"/g, "'"); 
						otherAttributes.push(`${attrName}="${safeValue}"`);
					}
				}
			});
			// --- 속성 처리 끝 ---

			// 괄호로 묶는 속성들 추가
			if (otherAttributes.length > 0) {
				attributesString += `(${otherAttributes.join(' ')})`;
			}


			// div 축약 로직은 그대로 유지
			let currentNode = node;
			while (
				currentNode.tagName === 'DIV' &&
				Array.from(currentNode.childNodes).filter(n => n.nodeType === Node.ELEMENT_NODE || n.nodeValue.trim()).length === 1 &&
				currentNode.firstElementChild?.tagName === 'DIV'
			) {
				currentNode = currentNode.firstElementChild;
			}

			// 태그 이름과 변환된 속성 문자열을 함께 추가
			lines.push(`${indent}${tagName}${attributesString}`);

			// textarea의 값 처리 (node.value 사용)
			if (tagName === 'textarea') {
				const value = node.value;
				if (value.trim()) {
					// 여러 줄 텍스트 처리를 위해 각 줄을 '| '로 시작
					value.split('\n').forEach(line => {
						lines.push(`${indent}    | ${line}`);
					});
				}
			}
			// 자식 노드 처리
			else if (currentNode.hasChildNodes()) {
				// textarea는 값 처리가 완료되었으므로, 자식 노드를 추가로 처리할 필요는 없습니다.
				// (일반적으로 textarea의 텍스트는 childNodes로도 잡히지만, value로 처리하는 것이 정확합니다.)
				if (tagName !== 'textarea') {
					lines = lines.concat(generatePugLines(currentNode.childNodes, indentLevel + 1));
				}
			}

		} else if (node.nodeType === Node.TEXT_NODE) {
			const textContent = node.nodeValue.trim();
			if (textContent) {
				lines.push(`${indent}| ${textContent}`);
			}
		}
	});

	return lines;
}

function safeClone(obj) {
	const seen = new WeakMap();
	function clone(value) {
		if (typeof value !== "object" || value === null) return value;
		if (seen.has(value)) return null; // 순환 참조 제거
		const copy = Array.isArray(value) ? [] : {};
		seen.set(value, copy);
		for (const key in value) {
			copy[key] = clone(value[key]);
		}
		return copy;
	}
	return clone(obj);
}



const twoPartDomains = ["co.kr","co.uk","co.jp","com.cn","co.in","com.mx","co.id","com.my","com.sg","com.ph","com.vn"];


// 국가 코드를 지역으로 매핑하는 맵
// 국가 코드를 지역으로 매핑하는 맵 (ISO 3166-1 alpha-2 기준)

/*
	logis 
		- pages 
		- tasks

	사용자 1000명씩 분할
		- vectorize, d1 둘다

	apac1-logis_items
	apac1-logis-goods
	apac1-logis-order
	apac1-logis-tracking
	apac1-logis-event

	...

*/ 


const CenterRegion = "center_logis"

const LogisRegion = {
	// Western North America
	'us-w': 'wnam_logis',
	'ca-w': 'wnam_logis',

	// Eastern North America
	'us': 'enam_logis',
	'ca': 'enam_logis',
	'mx': 'enam_logis',
	'cu': 'enam_logis',
	'do': 'enam_logis',
	'pr': 'enam_logis',
	'jm': 'enam_logis',

	// Western Europe
	'gb': 'weur_logis',
	'ie': 'weur_logis',
	'fr': 'weur_logis',
	'de': 'weur_logis',
	'nl': 'weur_logis',
	'be': 'weur_logis',
	'lu': 'weur_logis',
	'ch': 'weur_logis',
	'at': 'weur_logis',
	'es': 'weur_logis',
	'pt': 'weur_logis',
	'it': 'weur_logis',
	'se': 'weur_logis',
	'no': 'weur_logis',
	'dk': 'weur_logis',
	'fi': 'weur_logis',

	// Eastern Europe
	'ru': 'eeur_logis',
	'pl': 'eeur_logis',
	'cz': 'eeur_logis',
	'hu': 'eeur_logis',
	'ro': 'eeur_logis',
	'bg': 'eeur_logis',
	'ua': 'eeur_logis',
	'gr': 'eeur_logis',
	'rs': 'eeur_logis',

	// Asia_Pacific
	'cn': 'apac_logis',
	'hk': 'apac_logis',
	'kr': 'apac_logis',
	'jp': 'apac_logis',
	'sg': 'apac_logis',
	'tw': 'apac_logis',
	'th': 'apac_logis',
	'vn': 'apac_logis',
	'my': 'apac_logis',
	'ph': 'apac_logis',
	'id': 'apac_logis',
	'in': 'apac_logis',
	'pk': 'apac_logis',
	'bd': 'apac_logis',

	// Oceania
	'au': 'oc_logis',
	'nz': 'oc_logis',
	'fj': 'oc_logis',
	'pg': 'oc_logis',

	// South America
	'br': 'enam_logis', // Brazil
	'ar': 'enam_logis', // Argentina
	'cl': 'enam_logis', // Chile
	'co': 'enam_logis', // Colombia
	'pe': 'enam_logis', // Peru

	// Africa
	'za': 'weur_logis', // South Africa
	'ng': 'weur_logis', // Nigeria
	'eg': 'weur_logis', // Egypt

	// Middle East
	'sa': 'eeur_logis', // Saudi Arabia
	'ae': 'eeur_logis', // United Arab Emirates
	'tr': 'eeur_logis', // Turkey
};



const tables = ['items', 'sales', 'event', 'talks', 'tracking']


const Related = function(type){
	var list = []

	if(type == "goods"){
		list = ['order','tracking','coupon','event','member']

	}else if(type == "order"){
		list = ['goods','tracking','coupon','event','member']

	}else if(type == "tracking"){
		list = ['goods','order','coupon','event','member']

	}else if(type == "coupon"){
		list = ['goods','event','member']

	}else if(type == "event"){
		list = ['goods','coupon','member']

	}else if(type == "review"){
		list = ['goods','coupon','event','member']

	}

	return list
}


const parseStatus = function(status){
	var step = 0

	if(status == 'progress'){
		step = 1
	}else if(status == 'stop'){
		step = 2
	}else if(status == 'cancel'){
		step = 3
	}else if(status == 'refund'){
		step = 4
	}else if(status == 'return'){
		step = 5
	}else if(status == 'error'){
		step = 6
	}else if(status == 'expire'){
		step = 7
	}else if(status == 'exchange'){
		step = 8
	}else if(status == 'complete'){
		step = 9
	}else if(status == 'draft'){
		step = 10
	}else if(status == 'show'){
		step = 11
	}else if(status == 'hide'){
		step = 12
	}

	return step
}

const Relay = async function(foreign, primary){
	var query = []

	var merge = {}

	if(foreign == "goods" && primary.type == "order"){
		if(primary.tracking){
			query.push({
				type : primary.type,
				table : 'sales',
				column : 'tracking',
				value : primary.tracking
			})

			merge = {
				upsert : {
					includes : ["event", "width", "height", "length", "weight", "size", "currency", "cost_price", "sale_price", "discount", "quantity", "tracking", "number", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "fulfillment_service", "stock_keeping_unit", "bundle_shipping", "used", "lease", "rental", "refurbish", "tax_included", "release_date"],
					from : foreign,
					to : primary.type
				}
			}

		}else{
			query.push({
				type : primary.type,
				table : 'sales',
				column : 'index',
				value : primary.index
			})

			merge = {
				update : {
					includes : ["event", "width", "height", "length", "weight", "size", "currency", "cost_price", "sale_price", "discount", "quantity", "tracking", "number", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "fulfillment_service", "stock_keeping_unit", "bundle_shipping", "used", "lease", "rental", "refurbish", "tax_included", "release_date"],
					column : 'index',
					value : primary.index,
					from : foreign,
					to : primary.type
				}
			}
		}

	}else if(foreign == "tracking" && primary.type == "order"){
		// 단일 주문인경우 상품 제외하고 배송관련 내용만 업데이트함
		// 여러 상품 아이템은 tracking 번호만 업데이트함

		if(primary.tracking){
			query.push({
				type : foreign,
				table : 'tracking',
				column : primary.type,
				value : primary.index
			})

			merge = {
				update : {
					includes : ["width", "height", "length", "weight"],
					column : 'index',
					value : primary.index,
					foreign : {
						from : 'index',
						to : 'tracking',
					},
					from : primary.type,
					to : foreign
				}
			}
		}else{
			query.push({
				type : foreign,
				table : 'tracking',
				column : primary.type,
				value : primary.index
			})

			merge = {
				update : {
					includes : ["no", "goods", "event"],
					column : 'index',
					value : primary.index,
					foreign : {
						from : 'index',
						to : 'tracking',
					},
					from : foreign,
					to : primary.type
				}
			}
		}		

	}else if(foreign == "coupon" && primary.type == "order"){
		query.push({
			type : foreign,
			table : 'event',
			column : 'index',
			value : primary.event
		})

		merge = {
			update : {
				includes : ["discount"],
				column : 'index',
				value : primary.index,
				from : foreign,
				to : primary.type
			}
		}

	}else if(foreign == "event" && primary.type == "order"){
		query.push({
			type : foreign,
			table : 'event',
			column : 'index',
			value : primary.event
		})

		merge = {
			update : {
				includes : ["discount"],
				column : 'index',
				value : primary.index,
				from : foreign,
				to : primary.type
			}
		}



	}else if(foreign == "order" && primary.type == "goods"){
		query.push({
			type : foreign,
			table : 'sales',
			column : 'goods',
			value : primary.index
		})

		merge = {
			update : {
				includes : ["event", "width", "height", "length", "weight", "size", "currency", "cost_price", "sale_price", "discount", "quantity", "tracking", "number", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "fulfillment_service", "stock_keeping_unit", "bundle_shipping", "used", "lease", "rental", "refurbish", "tax_included", "release_date"],
				column : "goods",
				value : primary.index,
				from : primary.type,
				to : foreign
			}
		}
		
	}else if(foreign == "tracking" && primary.type == "goods"){
		// upsert goods 정보로 tracking 추가함
		query.push({
			type : "order",
			status : 0,
			table : 'tracking',
			column : 'goods',
			value : primary.index,
		})

		merge = {
			update : {
				includes : ["width", "height", "length", "weight", "shipping_fee", "shipping_method", "shipping_duration", "bundle_shipping"],
				from : primary.type,
				to : foreign
			}
		}

	}else if(foreign == "event" && primary.type == "goods"){
		query.push({
			type : foreign,
			table : 'event',
			column : 'index',
			value : primary.event
		})

		merge = {
			update : {
				includes : ["discount"],
				column : 'index',
				value : primary.index,
				from : foreign,
				to : primary.type
			}
		}

	}else if(foreign == "coupon" && primary.type == "goods"){
		query.push({
			type : foreign,
			table : 'event',
			column : 'index',
			value : primary.event
		})

		merge = {
			update : {
				includes : ["discount"],
				column : 'index',
				value : primary.index,
				from : foreign,
				to : primary.type
			}
		}



	}else if(foreign == "goods" && primary.type == "tracking"){
		// upsert goods 정보로 tracking 추가함
		query.push({
			type : "order",
			status : 0,
			table : 'sales',
			column : "goods",
			value : primary.goods
		})

		merge = {
			update : {
				includes : ["width", "height", "length", "weight", "shipping_fee", "shipping_method", "shipping_duration", "bundle_shipping"],
				column : 'index',
				value : primary.index,
				from : foreign,
				to : primary.type
			}
		}

	}else if(foreign == "order" && primary.type == "tracking"){
		if(primary.goods){
			query.push({
				type : foreign,
				table : 'sales',
				column : 'goods',
				value : primary.goods
			})

			merge = {
				update : {
					includes : ["width", "height", "length", "weight", "shipping_fee", "shipping_method", "shipping_duration", "bundle_shipping"],
					column : 'tracking',
					value : primary.index,
					foreign : {
						from : 'index',
						to : 'tracking',
					},	
					from : primary.type,
					to : foreign
				}
			}
		}else{
			query.push({
				type : foreign,
				table : 'tracking',
				column : primary.type,
				value : primary.index
			})

			merge = {
				update : {
					includes : ["no", "order", "goods", "event"],
					column : 'index',
					value : primary.index,
					foreign : {
						from : 'index',
						to : 'order',
					},
					from : foreign,
					to : primary.type
				}
			}
		}

	}else if(foreign == "event" && primary.type == "tracking"){
		// 매칭이 아예 안되는 항목

	}else if(foreign == "coupon" && primary.type == "tracking"){
		// 매칭이 아예 안되는 항목



	}else if(foreign == "goods" && primary.type == "event"){
		query.push({
			type : foreign,
			table : 'sales',
			column : 'event',
			value : primary.index
		})

		merge = {
			update : {
				includes : ["discount"],
				column : 'event',
				value : primary.index,
				from : primary.type,
				to : foreign
			}
		}

	}else if(foreign == "order" && primary.type == "event"){
		query.push({
			type : foreign,
			table : 'sales',
			column : 'event',
			value : primary.index
		})

		merge = {
			update : {
				includes : ["discount"],
				column : 'event',
				value : primary.index,
				from : primary.type,
				to : foreign
			}
		}

	}else if(foreign == "tracking" && primary.type == "event"){
		// 매칭이 아예 안되는 항목

	}else if(foreign == "coupon" && primary.type == "event"){
		query.push({
			type : foreign,
			table : 'event',
			column : 'event',
			value : primary.index
		})

		merge = {
			update : {
				includes : ["started_at", "expired_at", "phone", "address", "discount", "quantity", "usage_per", "usage_limit", "min_order_amount", "max_order_amount", "max_discount_amount", "new_customer_only", "first_purchase_only", "region_restrictions"],
				column : 'event',
				value : primary.index,
				from : primary.type,
				to : foreign
			}
		}



	}else if(foreign == "goods" && primary.type == "coupon"){
		query.push({
			type : foreign,
			table : 'sales',
			column : 'event',
			value : primary.index
		})

		merge = {
			from : primary.type,
			to : foreign
		}

	}else if(foreign == "order" && primary.type == "coupon"){
		query.push({
			type : foreign,
			status : 0,
			table : 'sales',
			column : 'event',
			value : primary.index
		})

		merge = {
			update : {
				includes : ["discount"],
				column : 'event',
				value : primary.index,
				from : primary.type,
				to : foreign
			}
		}

	}else if(foreign == "tracking" && primary.type == "coupon"){
		// 매칭이 아예 안되는 항목

	}else if(foreign == "event" && primary.type == "coupon"){
		if(typeof primary.event != "undefined"){
			query.push({
				type : foreign,
				table : 'event',
				column : 'index',
				value : primary.event
			})

			merge = {
				update : {
					includes : ["started_at", "expired_at", "phone", "address", "discount", "quantity", "usage_per", "usage_limit", "min_order_amount", "max_order_amount", "max_discount_amount", "new_customer_only", "first_purchase_only", "region_restrictions"],
					column : 'index',
					value : primary.index,
					from : foreign,
					to : primary.type
				}
			}
		}
	}

	return {
		query : query,
		merge : merge
	}
}




/*
	벡터맵으로 구분하자
	wnam-logis      Western North America
	enam-logis      Eastern North America
	weur-logis      Western Europe
	eeur-logis      Eastern Europe
	apac-logis      Asia-Pacific
	oc-logis            Oceania


*/ 


const Hello = {
	"Korean": "안녕하세요 내용을 입력해주세요",
	"Japanese": "こんにちは、内容を入力してください",
	"English": "Hello, please enter the content",
	"Chinese": "你好，请输入内容",
	"French": "Bonjour, veuillez saisir le contenu",
	"German": "Hallo, bitte geben Sie den Inhalt ein",
	"Spanish": "Hola, por favor ingrese el contenido",
	"Russian": "Здравствуйте, пожалуйста, введите содержание",
	"Arabic": "مرحبًا، يرجى إدخال المحتوى"
}

const languageCodeToCountryCode = {
	'ko': 'kr', // Korean -> South Korea
	'ja': 'jp', // Japanese -> Japan
	'en': 'us', // English -> United States (가장 일반적인 영어를 사용하는 국가)
	'zh': 'cn', // Chinese -> China (가장 일반적인 중국어를 사용하는 국가)
	'fr': 'fr', // French -> France
	'de': 'de', // German -> Germany
	'es': 'es', // Spanish -> Spain
	'ru': 'ru', // Russian -> Russia
	'ar': 'sa', // Arabic -> Saudi Arabia
};


const languageCode = {
	// Western North America
	'us-w': 'English',
	'ca-w': 'English',

	// Eastern North America
	'us': 'English',
	'ca': 'English',
	'mx': 'Spanish',
	'cu': 'Spanish',
	'do': 'Spanish',
	'pr': 'Spanish',
	'jm': 'English',

	// Western Europe
	'gb': 'English',
	'ie': 'English',
	'fr': 'French',
	'de': 'German',
	'nl': 'English',
	'be': 'French',
	'lu': 'French',
	'ch': 'German',
	'at': 'German',
	'es': 'Spanish',
	'pt': 'Portuguese',
	'it': 'Italian',
	'se': 'Swedish',
	'no': 'Norwegian',
	'dk': 'Danish',
	'fi': 'Finnish',

	// Eastern Europe
	'ru': 'Russian',
	'pl': 'Polish',
	'cz': 'Czech',
	'hu': 'Hungarian',
	'ro': 'Romanian',
	'bg': 'Bulgarian',
	'ua': 'Ukrainian',
	'gr': 'Greek',
	'rs': 'Serbian',

	// Asia-Pacific
	'cn': 'Simplified Chinese',
	'hk': 'Traditional Chinese',
	'kr': 'Korean',
	'jp': 'Japanese',
	'sg': 'English',
	'tw': 'Traditional Chinese',
	'th': 'Thai',
	'vn': 'Vietnamese',
	'my': 'Malay',
	'ph': 'English',
	'id': 'Indonesian',
	'in': 'English',
	'pk': 'Urdu',
	'bd': 'Bengali',

	// Oceania
	'au': 'English',
	'nz': 'English',
	'fj': 'English',
	'pg': 'English',

	// South America
	'br': 'Portuguese', // Brazil
	'ar': 'Spanish', // Argentina
	'cl': 'Spanish', // Chile
	'co': 'Spanish', // Colombia
	'pe': 'Spanish', // Peru

	// Africa
	'za': 'English', // South Africa
	'ng': 'English', // Nigeria
	'eg': 'Arabic', // Egypt

	// Middle East
	'sa': 'Arabic', // Saudi Arabia
	'ae': 'Arabic', // United Arab Emirates
	'tr': 'Turkish' // Turkey
}

function parseBody(body, page){
	var body = ''

	// pug를 html로 변경하고 body안에 값 넣어야함 그래야 돌아감

	var { document } = parseHTML(`<html><body>${body}</body></html>`);

	var results = []

	for (const s in page.selectors) {
		if (selectors.hasOwnProperty(s)) {
			var selector = selectors[s]

			var item = {}

			var $item = document.querySelector(selector)

			if($item){
				var type = $item.getAttribute('type')

				var checked = $item.getAttribute('checked')

				var selected = $item.getAttribute('selected')

				if(type){
					var text = $item.textContent

					if(checked){
						item[s] = checked == "true" ? true : false
					}else if(selected){
						item[s] = selected
					}else if($item.value){
						item[s] = $item.value
					}else if($item.textContent){
						item[s] = $item.textContent     
					}else{
						item[s] = null
					}
				}else{
					item[s] = $item.textContent ? $item.textContent : null  
				}
			}

			results.push(item)
		}
	}

	return results
}


async function arrayBufferToBase64(arrayBuffer) {
	const bytes = new Uint8Array(arrayBuffer)

	let binary = ''

	for (let i = 0; i < bytes.byteLength; i++) {
		binary += String.fromCharCode(bytes[i])
	}

	return btoa(binary)
}

async function Deepinfra(key, model, system, user, inlineData){
	// DeepInfra API 호출

	var messages = []

	if(inlineData){
		messages.push({
			type: "image_url",   // 여기서 URL 입력
			image_url: {
				url: inlineData.data,
				detail: "auto"
			}
		})
	}

	if(system){
		messages.push({ "role": "system", "content": system })
	}

	if(user){
		messages.push({ "role": "user", "content": user })
	}

	
		
	var body = {
		"model" : model,
		"messages": messages,
		"max_tokens": 5000,
		"temperature": 1
	}

	var pathname = 'chat/completions'

	var isEmbedding = model.indexOf('BAAI/bge-m3') > -1

	if(isEmbedding){
		pathname = 'embeddings'

		body = {
			"input": system + user,
			"model": model,
			"encoding_format": "float"
		}
	}

	var res = await fetch(`https://api.deepinfra.com/v1/openai/${pathname}`, {
		method: "POST",
		headers: {
			"Authorization": `Bearer ${key}`,
			"Content-Type": "application/json",
		},
		body: JSON.stringify(body),
	});

	var json = await res.json();



	if(isEmbedding){
		return json.data[0].embedding
	}else{
		var content = json.choices[0].message.content;

		console.log('content',content);

		try{
			if(content.indexOf('```') > -1){
				content = content.replace(/```json/gi, "")
				content = content.replace(/```/gi, "")
				content = content.replace(/\n/gi,"")
				content = content.trim()
			}

			var results = JSON.parse(content)

			return safeClone(results)
		}catch(err){
			return content
		}
	}
}

async function Gemini(key, model, system, user, config, inlineData){
	console.log('Gemini 진입');

	if(typeof config == "undefined"){
		config = {
			"response_mime_type": "application/json",
			"temperature": 1
		}
	}

	var parts = [{
		text: system + user
	}]

	if(inlineData){
		parts.push({ inlineData: inlineData })
	}

	var res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
		},
		body: JSON.stringify({
			contents: [{
				parts: parts
			}],
			generationConfig: config
		})
	})

	var data = await res.json()

	var content = data.candidates[0].content.parts[0].text

	if(config["response_mime_type"]){
		try{
			if(content.indexOf('```') > -1){
				content = content.replace(/```json/gi, "")
				content = content.replace(/```/gi, "")
				content = content.replace(/\n/gi,"")
				content = content.trim()    
			}

			var results = JSON.parse(content)


			return safeClone(results.length ? results[0] : results)
		}catch(err){
			
		}
	}

	return content
}


/*
	cf/google/embeddinggemma-300m
	google/embeddinggemma-300m
*/


export default {
	async fetch(
		request: Request,
		env: Env,
		ctx: ExecutionContext
	): Promise<Response> {
		// task 실행

		try{
			const buffer = await request.arrayBuffer()

			console.log('buffer.byteLength',buffer.byteLength);

			if(buffer.byteLength){
				var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(buffer))

				var json = JSON.parse(decompressedJsonString)

				var now = json.now

				var gemini_llm_api = json.gemini_llm_api

				var gemini_llm_model = json.gemini_llm_model

				var deepinfra = json.deepinfra

				var current = new Date(now).toISOString()

				var created_at = now - 10000

				var pageCount = json.counts

				var limits = json.limits

				var models = json.models

				var region = json.region

				console.log('region',region);

				// console.log('json',JSON.stringify(json));

				var fallback = ''

				var { results } = await env[region].prepare(`SELECT * FROM tasks WHERE "ref" = "${json.ref}" AND "created_at" < ${created_at} AND "updated_at" = 0 ORDER BY created_at ASC LIMIT 1`).all()

				console.log('results.length',results.length);

				var crons = safeClone(results)

				if(crons.length){
					for(var c = 0; c < crons.length; c++){
						var cron = crons[c]

						var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(cron.task))

						var task = JSON.parse(decompressedJsonString)

						var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
							text : task.text
						})), { to: 'arraybuffer' })

						task.data = arr.buffer

						var talk = {
							id : task.id,
							type : task.type,
							from : task.from,
							to : task.to,
							cc : task.cc,
							bcc : task.bcc,
							ref : task.ref,
							data : task.data,
							created_at : now,
							updated_at : now
						}

						var logisRegion = LogisRegion[task.flag]

						var zoneRegion = task.zone

						var vectorRegion = zoneRegion.replace(/_/gi,"-")

						var language = languageCode[task.flag]

						if(!models[task.cc]){
							models[task.cc] = task.rpm
						}

						if(models[task.cc]){
							models[task.cc] -= 1
						}else{
							fallback = 'models'

							continue;
						}

						if(!limits[task.from]){
							limits[task.from] = task.rpm
						}

						// 팀 계정으로 해야함
						var { results } = await env[logisRegion].prepare(`SELECT * FROM users WHERE "type" = "team" AND "id" = "${task.team}" AND "created_at" < ${now} LIMIT 1`).all()

						var team = results[0]

						if(team.data){
							var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(team.data))

							team.data = JSON.parse(decompressedJsonString)
						}else{
							team.data = {}
						}


						var statements = {}
							statements[CenterRegion] = []

						if(!statements[logisRegion]){
							statements[logisRegion] = []
						}

						if(!statements[`${zoneRegion}-${tables[0]}`]){
							for(var t = 0; t < tables.length; t++){
								var table = tables[t]

								if(!statements[`${zoneRegion}_${table}`]){
									statements[`${zoneRegion}_${table}`] = []
								}
							}
						}


						// 오픈 하기전에 반영해야함

						// if(limits[team.id]){
						// 	limits[team.id] -= 1
						// }else{
						// 	if(typeof limits[team.id] == "undefined"){
						// 		limits[team.id] = 0
						// 	}else{
						// 		fallback = 'out of gas'

						// 		continue
						// 	}
						// }


						// model context protocol

						if(task.contentType == "image/jpeg"){
							var base64 = arrayBufferToBase64(task.buffer)

							var inlineData = { mimeType: task.contentType, data: base64 }

							var type = talk.type = task.type


							// 주소 조회해야함
							// 겸사 겸사 업체 정보 등록받자
							var address = team.data.address ? team.data.address : []

							var system = image2json(type, address)

							var content = task.text


							var item

							if(models['deepinfra']){
								item = await Deepinfra(deepinfra, 'google/gemma-3-27b-it', system, content, inlineData)

								models['deepinfra'] -= 1

							}

							if(!item && gemini_llm_api){
								item = await Gemini(gemini_llm_api, gemini_llm_model, system, content, null, inlineData)

								models[gemini_llm_api+'-'+gemini_llm_model] -= 1

							}

							if(!item){
								fallback = 'overflow'

								continue
							}

							if(!item.id){
								fallback = 'ShippingLabel Not Found'
								// 올바르지 않은 이미지 안내하기

								continue
							}


							talk.text = item.text


							var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify(item)), { to: 'arraybuffer' })

							item.data = arr.buffer



							/*
								입고, 출고 나가면 증가 및 차감하는 로직 반영해야함

								업체 주소 미리 입력되어있고, 받는 사람에 LLM으로 true 시 입고, false시 출고



								item.no == item.index 먼저 조회하고 없으면
								
								barcode 찾는 형식으로 해야함

								둘다 없으면 type 'draft'로 전부 추가해야함
							*/
							

							item.no = item.id

							if(item.no.indexOf("-") > -1){
								item.no = item.no.replace(/-/gi,"")
							}

							if(item.no.indexOf("_") > -1){
								item.no = item.no.replace(/_/gi,"")
							}

							item.id = hashId(team.id+item.no)

							item.type = type

							item.from = task.from

							item.to = task.to

							item.cc = task.cc // logis.center로 잡혀져 있음 

							item.bcc = task.bcc

							item.ref = task.ref

							item.created_at = now

							item.index = crc32(team.id+item.id)


							var { results } = await env[`${zoneRegion}_sales`].prepare(`SELECT * FROM sales WHERE "tracking" = ${item.index} AND "to" = "${task.to}" AND "created_at" < ${now} LIMIT 1`).all()

							var sales = results

							if(results.length){
								var _item = safeClone(results[0])

								delete _item.id
								delete _item.type
								delete _item.from
								delete _item.to
								delete _item.cc
								delete _item.bcc
								delete _item.data
								delete _item.created_at

								item = mergeNode(item, _item)
							}


							if(type == "tracking"){
								var { results } = await env[`${zoneRegion}_tracking`].prepare(`SELECT * FROM tracking WHERE "id" = "${item.id}" AND "to" = "${task.to}" AND "created_at" < ${now} LIMIT 1`).all()

								if(results.length){
									var _item = results[0]

									item = mergeNode(item, _item)


									if(sales.length){
										var _item = sales[0]

										statements[`${zoneRegion}_sales`].push(
											env[`${zoneRegion}_sales`].prepare(`
												UPDATE sales SET updated_at = ?, status = ? WHERE id = ?
											`).bind(
												now, _item.status, _item.id
											)
										)
									}
								}



								if(sales.length){
									var _item = sales[0]

									statements[`${zoneRegion}_items`].push(
										env[`${zoneRegion}_items`].prepare(`
											INSERT INTO items (
												"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "updated_at"
											) VALUES (
												?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10
											) ON CONFLICT (id) DO UPDATE SET
												"type" = EXCLUDED."type",
												"from" = EXCLUDED."from",
												"to" = EXCLUDED."to",
												"cc" = EXCLUDED."cc",
												"bcc" = EXCLUDED."bcc",
												"ref" = EXCLUDED."ref",
												"data" = EXCLUDED."data",
												"created_at" = EXCLUDED."created_at",
												"updated_at" = EXCLUDED."updated_at"
										`).bind(
											_item.id,
											_item.type,
											_item.from,
											_item.to,
											_item.cc,
											_item.bcc,
											_item.data,
											item.id,
											_item.created_at,
											now
										)
									)
								}else{
									var content = {}

									if(item.title){
										content.title = item.title
									}

									if(item.sender_address){
										content.sender_address = item.sender_address
									}

									if(item.recipient_address){
										content.recipient_address = item.recipient_address
									}

									if(item.carrier){
										content.carrier = item.carrier
									}

									if(item.shipping_method){
										content.shipping_method = item.shipping_method
									}

									if(item.fulfillment_service){
										content.fulfillment_service = item.fulfillment_service
									}

									item.type = item.recipient_match ? 'receiving' : 'shipping'

									var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
										type : item.type,
										text : item.text,
										link : null
									})), { to: 'arraybuffer' })

									item.data = arr.buffer


									var metadata = {
										id: item.id,
										type: item.type,
										from: task.from,
										to: task.to,
										cc: task.cc,
										bcc: task.bcc,
										ref:task.ref
									}

									var embeddings

									if(models['cloudflare']){
										var { data: embeddings } = await env.AI.run('@cf/baai/bge-m3', {
											text: [item.text]
										})

										var $VectorizeVector = [
											{
												id: item.id,
												values: embeddings[0],
												metadata: metadata
											}
										]

										models['cloudflare'] -= 1

									}

									if(!embeddings && models['deepinfra']){
										var embeddings = await Deepinfra(deepinfra, 'BAAI/bge-m3', '', item.text)

										var $VectorizeVector: VectorizeVector[] = embeddings.map((values, i) => {
											return {
												id: item.id,
												values: values,
												metadata: metadata
											}
										})

										models['deepinfra'] -= 1

									}

									if(!embeddings){
										fallback = 'overflow'

										continue
									}

									await env[`${vectorRegion}-${type}`].upsert($VectorizeVector)

									statements[`${zoneRegion}_items`].push(
										env[`${zoneRegion}_items`].prepare(`
											INSERT INTO items (
												"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "updated_at"
											) VALUES (
												?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10
											) ON CONFLICT (id) DO UPDATE SET
												"type" = EXCLUDED."type",
												"from" = EXCLUDED."from",
												"to" = EXCLUDED."to",
												"cc" = EXCLUDED."cc",
												"bcc" = EXCLUDED."bcc",
												"ref" = EXCLUDED."ref",
												"data" = EXCLUDED."data",
												"created_at" = EXCLUDED."created_at",
												"updated_at" = EXCLUDED."updated_at"
										`).bind(
											task.id,
											task.type,
											task.from,
											task.to,
											task.cc,
											task.bcc,
											task.ref,
											arr.buffer,
											now,
											0
										)
									)
								}


								statements[`${zoneRegion}_tracking`].push(
									env[`${zoneRegion}_tracking`].prepare(`
										INSERT INTO tracking (
											"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "index", "event", "goods", "order", "status", "no", "sender_address", "sender_phone", "recipient_address", "recipient_phone", "width", "height", "length", "weight", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "shipping_date", "delivery_date", "order_date", "payment_date", "payment_method", "payment_origin", "payment_number", "bundle_shipping"
										) VALUES (
											?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31, ?32, ?33, ?34, ?35
										) ON CONFLICT (id) DO UPDATE SET
											"type" = EXCLUDED."type",
											"from" = EXCLUDED."from",
											"to" = EXCLUDED."to",
											"cc" = EXCLUDED."cc",
											"bcc" = EXCLUDED."bcc",
											"ref" = EXCLUDED."ref",
											"data" = EXCLUDED."data",
											"created_at" = EXCLUDED."created_at",
											"index" = EXCLUDED."index",
											"event" = EXCLUDED."event", 
											"goods" = EXCLUDED."goods", 
											"order" = EXCLUDED."order",
											"status" = EXCLUDED."status",
											"no" = EXCLUDED."no",
											"sender_address" = EXCLUDED."sender_address",
											"sender_phone" = EXCLUDED."sender_phone",
											"recipient_address" = EXCLUDED."recipient_address",
											"recipient_phone" = EXCLUDED."recipient_phone",
											"width" = EXCLUDED."width",
											"height" = EXCLUDED."height",
											"length" = EXCLUDED."length",
											"weight" = EXCLUDED."weight",
											"carrier" = EXCLUDED."carrier",
											"shipping_fee" = EXCLUDED."shipping_fee",
											"shipping_method" = EXCLUDED."shipping_method",
											"shipping_duration" = EXCLUDED."shipping_duration",
											"shipping_date" = EXCLUDED."shipping_date",
											"delivery_date" = EXCLUDED."delivery_date",
											"order_date" = EXCLUDED."order_date",
											"payment_date" = EXCLUDED."payment_date",
											"payment_method" = EXCLUDED."payment_method",
											"payment_origin" = EXCLUDED."payment_origin",
											"payment_number" = EXCLUDED."payment_number",
											"bundle_shipping" = EXCLUDED."bundle_shipping"
									`).bind(
										item.id,
										item.type,
										item.from,
										item.to,
										item.cc,
										item.bcc,
										item.ref,
										item.data,
										item.created_at,
										item.index,
										item.event ? item.event : 0,
										item.goods ? item.goods : 0,
										item.order ? item.order : 0,
										parseStatus(item.status),
										item.no,
										item.sender_address ? item.sender_address : "",
										item.sender_phone ? item.sender_phone : "",
										item.recipient_address ? item.recipient_address : "",
										item.recipient_phone ? item.recipient_phone : "",
										item.width ? parseFloat(item.width) : 0,
										item.height ? parseFloat(item.height) : 0,
										item.length ? parseFloat(item.length) : 0,
										item.weight ? parseFloat(item.weight) : 0,
										item.carrier ? parseFloat(item.carrier) : 0,
										item.shipping_fee ? parseFloat(item.shipping_fee) : 0,
										item.shipping_method ? item.shipping_method : "",
										item.shipping_duration ? parseFloat(item.shipping_duration) : 0,
										item.shipping_date ? parseFloat(item.shipping_date) : 0,
										item.delivery_date ? parseFloat(item.delivery_date) : 0,
										item.order_date ? parseFloat(item.order_date) : 0,
										item.payment_date ? parseFloat(item.payment_date) : 0,
										item.payment_method ? item.payment_method : "",
										item.payment_origin ? item.payment_origin : "",
										item.payment_number ? item.payment_number : "",
										item.bundle_shipping ? parseFloat(item.bundle_shipping) : 0
									)
								)
							}
								

						}else if(task.scan){
							// INSERT 백터 생성 INSERT

							var isDetail = false

							try{
								var page

								var pageType = ''

								var pageLength = 0

								var url = new URL(task.href)

								var pageId = hashId(task.cc+url.pathname)

								var { results } = await env[CenterRegion].prepare(`SELECT * FROM pages WHERE "id" = "${pageId}" AND "created_at" < ${created_at} LIMIT 1`).all()

								if(results.length){
									var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(results[0].data))

									var selectors = JSON.parse(decompressedJsonString)

									if(selectors.type && selectors.node){
										try{
											var { document } = parseHTML(`<html><body>${task.text}</body></html>`);
											
											if(document.querySelector(selectors.node)){
												task.text = document.querySelector(selectors.node).innerHTML
											}
										}catch(err){
											console.log('page err',err);
										}
									}
								}


								// 기존 값이 있으면 아래 프로세스 실행함

								var itemId = hashId(team.id+task.cc+task.link)

								var { results } = await env[`${zoneRegion}_items`].prepare(`SELECT * FROM items WHERE "id" = "${itemId}" AND "created_at" < ${created_at} LIMIT 1`).all()

								if(results.length){
									if(task.ref){
										var { results } = await env[CenterRegion].prepare(`SELECT * FROM pages WHERE "id" = "${task.ref}" AND "created_at" < ${created_at} LIMIT 1`).all()

										if(results.length){
											var _page = results[0]

											if(_page.type){
												var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(_page.data))

												var _data = JSON.parse(decompressedJsonString)

												if(_data.item){
													isDetail = true

													pageType = _page.type
												}
											}
										}
									}
								}

								var content = convertHtmlToCleanPug(task.text)

								if(!isDetail){
									var system = list2json(language)

									system = system.trim()

									if(models['deepinfra']){
										page = await Deepinfra(deepinfra, 'openai/gpt-oss-20b', `Analyze the provided Pug template and return it in the following JSON format, no explanation. {language:'${language}',${system}}`, content)

										pageType = page.type

										if(page.items){
											pageLength = page.items.length
										}

										isDetail = page.isDetail

										models['deepinfra'] -= 1

									}

									if(!page && gemini_llm_api){
										page = await Gemini(gemini_llm_api, gemini_llm_model, `Analyze the provided Pug template and return it in the following JSON format, no explanation. {language:'${language}',${system}}`, content)

										pageType = page.type

										if(page.items){
											pageLength = page.items.length
										}

										isDetail = page.isDetail

										models[gemini_llm_api+'-'+gemini_llm_model] -= 1

									}
								}


								if((!isDetail && !pageLength) || isDetail){
									var system = item2json(pageType)

									system = system.trim()

									if(models['deepinfra']){
										page = await Deepinfra(deepinfra, 'openai/gpt-oss-20b', `Analyze the provided Pug template and return it in the following JSON format, no explanation. {language:'${language}',${system}}`, content)

										models['deepinfra'] -= 1

									}

									if(!page && gemini_llm_api){
										page = await Gemini(gemini_llm_api, gemini_llm_model, `Analyze the provided Pug template and return it in the following JSON format, no explanation. {language:'${language}',${system}}`, content)

										models[gemini_llm_api+'-'+gemini_llm_model] -= 1

									}
								}

								

								if(!page){
									fallback = 'page overflow'

									continue
								}

								page.type = pageType
								page.from = task.from
								page.to = task.to
								page.cc = task.cc
								page.bcc = task.bcc

								console.log('page', JSON.stringify(page));

								console.log('isDetail', JSON.stringify(isDetail));

								if(isDetail){
									page.items = [safeClone(page)]
								}


								page.id = pageId


								talk.text = page.text


								var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
									text : page.text || "",
									node : page.node || "",
									item : page.item || "",
									more : page.more || "",
									next : page.next || ""
								})), { to: 'arraybuffer' })

								page.ref = task.ref

								page.data = arr.buffer

								statements[CenterRegion].push(
									env[CenterRegion].prepare(`
										INSERT INTO pages ("id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "updated_at")
										VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
										ON CONFLICT(id) DO UPDATE SET
											"type" = EXCLUDED."type",
											"from" = EXCLUDED."from",
											"to" = EXCLUDED."to",
											"cc" = EXCLUDED."cc",
											"bcc" = EXCLUDED."bcc",
											"ref" = EXCLUDED."ref",
											"data" = EXCLUDED."data",
											"created_at" = EXCLUDED."created_at",
											"updated_at" = EXCLUDED."updated_at"
									`).bind(
										page.id,
										page.type,
										page.from,
										page.to,
										page.cc,
										page.bcc,
										page.ref,
										page.data,
										now,
										now
									)
								)


								talk.type = page.type

								/*
									items에서 
										시작 시간
										최저가 최대가격
										최저 수량 최대 수량

										값 가져오기
								*/ 

								var items = page.items ? page.items : []

								console.log('items',JSON.stringify(items));

								if(items.length){
									/*
										주문이후의 절차는 주문번호로 매칭해야함

										type : tracking 	// 배송추적
															// "고객 주문"" or "자사 재고" 등으로 추상화 매칭

										type : order
											order 파생 정보는
											전체 주문 목록 스캔하여
											주문 아이템 링크 클릭시
											레퍼러 참조 이벤트 추적하여 기록
											이 부분 크롬 익스텐션에서 해야함


											이 부분은 무조건 유료만 가능하게

												벡터 쿼리로 미리 저장하고 
													type : order, semantic : cancel     // 주문취소
													type : order, semantic : exchange   // 교환
													type : order, semantic : return     // 반품
													type : order, semantic : refund     // 환불

												title값 쿼리로 semantic 선택

												var { data: queryVector } = await env.AI.run('@cf/baai/bge-m3', {
													text: [semantic],
												})

												var { matches } = await env.SEMANTIC.query(queryVector[0], query.options)

									*/

									for(var i = 0; i < items.length; i++){
										var item = items[i]

										if(item.date){
											item.started_at = new Date(item.date).getTime()
										}

										if(!item.id){
											continue
										}

										if(isDetail){
											item.link = task.link
										}

										item.type = page.type

										item.no = (item.id ? item.id : i).toString()

										if(item.no.indexOf("-") > -1){
											item.no = item.no.replace(/-/gi,"")
										}

										if(item.no.indexOf("_") > -1){
											item.no = item.no.replace(/_/gi,"")
										}

										item.index = crc32(hashId(team.id+item.no))

										try{
											try{
												var url = new URL(item.link)
											}catch(err){
												var url = new URL(task.origin+item.link)

												item.link = url.pathname + url.search
											}

											item.link = url.pathname + url.search

										}catch(err){

										}


										var itemType = item.type

										if(item.type == "sales"){
											itemType = "sales"
											item.type = "order"

										}else if(item.type == "goods" || item.type == "order"){
											itemType = "sales"

										}else if(item.type == "event" || item.type == "coupon"){
											itemType = "event"

										}


										item.id = hashId(team.id+task.cc+item.link)


										if(item.type == "tracking"){
											var { results } = await env[`${zoneRegion}_tracking`].prepare(`SELECT * FROM tracking WHERE index" = "${item.index}" AND "to" = "${task.to}" AND "created_at" < ${now} LIMIT 1`).all()

											if(results.length){
												var _item = results[0]

												item = mergeNode(item, _item)
											}

											item.id = hashId(team.id+task.cc+task.link)
										}

										item.flag = task.flag
										
										item.from = task.from
										item.to = task.to
										item.cc = task.cc
										item.bcc = task.bcc

										item.ref = task.ref


										var goods = item.goods ? safeClone(item.goods) : []

										delete item.goods

										if(typeof goods.length != "undefined"){
											goods.unshift({})
										}else{
											goods = []
										}


										item.currency = item.currency ? item.currency.toUpperCase() : ""

										item.quantity = item.quantity ? parseInt(item.quantity) : 0

										item.created_at = now

										item.updated_at = now

										item.semantic = item.title

										item.started_at = item.manufacture_date ? item.manufacture_date : 0
										
										item.expired_at = item.expiration_date ? item.expiration_date : 0

										


										var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
											id : item.id,
											title : item.title,
											link : item.link,
											origin : task.origin ? task.origin : ''
										})), { to: 'arraybuffer' })

										item.data = arr.buffer

										if(item.tracking_number){
											var tracking_number = item.tracking_number

											if(tracking_number.indexOf("-") > -1){
												tracking_number = tracking_number.replace(/-/gi,"")
											}

											if(tracking_number.indexOf("_") > -1){
												tracking_number = tracking_number.replace(/_/gi,"")
											}

											item.tracking = crc32(hashId(team.id+tracking_number))
										}

											

										try{
											console.log('item.type',item.type);
											console.log('진입',item.tracking_number);

											if(item.type == "order" && item.tracking_number){
												/*
													상세와 리스트 차이가 분명히 있음

													주문 
														리스트에서는 송장번호가 없음
														상세페이지에서는 송장번호가 있음
												*/

											

												console.log('before item.tracking',item.tracking);

												if(goods.length){
													for(var g = 0; g < goods.length; g++){
														var good = safeClone(goods[g])

														var tracking = safeClone(item)

														tracking.type = "tracking"

														tracking.no = tracking_number

														tracking.index = item.tracking

														if(good.id){
															var no = good.id.toString()

															if(no.indexOf("-") > -1){
																no = no.replace(/-/gi,"")
															}

															if(no.indexOf("_") > -1){
																no = no.replace(/_/gi,"")
															}

															good.no = no

															good.index = crc32(hashId(team.id+good.no))

															var { results } = await env[`${zoneRegion}_sales`].prepare(`SELECT * FROM sales WHERE "type" = "goods" AND "index" = "${good.index}" AND "to" = "${task.to}" AND "created_at" < ${now} LIMIT 1`).all()

															if(results.length){
																tracking.event = results[0].event
															}

															tracking.goods = good.index

															tracking.id = hashId(team.id+good.no)
														}else{
															tracking.id = hashId(team.id+tracking.no)
														}



														tracking.order = item.index

														tracking.order_date = item.order_date
														tracking.payment_date = item.payment_date
														tracking.payment_method = item.payment_method
														tracking.payment_origin = item.payment_origin

														tracking.link = item.link

														tracking.sender_address = item.sender_address
														tracking.sender_phone = item.sender_phone
														tracking.recipient_address = item.recipient_address
														tracking.recipient_phone = item.recipient_phone

														tracking.data = {
															id : item.id,
															link : item.link,
															origin : task.origin ? task.origin : ""
														}



														var { results } = await env[`${zoneRegion}_tracking`].prepare(`SELECT * FROM tracking WHERE "index" = "${tracking.index}" AND "to" = "${task.to}" AND "created_at" < ${now} LIMIT 1`).all()

														if(results.length){
															var _tracking = safeClone(results[0])

															var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(_tracking.data))

															_tracking.data = JSON.parse(decompressedJsonString)

															tracking = mergeNode(_tracking, tracking)
														}else{
															// 처음 저장할때 자연어 LLM으로 전처리해서 벡터 저장해야함

															// var metadata = {
															// 	type: item.type,
															// 	from: item.from,
															// 	to: item.to,
															// 	cc: item.cc,
															// 	bcc: item.bcc,
															// 	ref:task.ref
															// }

															// var embeddings

															// if(models['cloudflare']){
															// 	var { data: embeddings } = await env.AI.run('@cf/baai/bge-m3', {
															// 		text: [item.semantic]
															// 	})

															// 	var $VectorizeVector = [
															// 		{
															// 			id: item.id,
															// 			values: embeddings[0],
															// 			metadata: metadata
															// 		}
															// 	]

															// 	models['cloudflare'] -= 1

															// }

															// if(!embeddings && models['deepinfra']){
															// 	var embeddings = await Deepinfra(deepinfra, 'BAAI/bge-m3', '', item.semantic)

															// 	var $VectorizeVector: VectorizeVector[] = embeddings.map((values, i) => {
															// 		return {
															// 			id: item.id,
															// 			values: values,
															// 			metadata: metadata
															// 		}
															// 	})

															// 	models['deepinfra'] -= 1
															// }

															// console.log('typeof embeddings',typeof embeddings);

															// if(!embeddings){
															// 	fallback = 'embeddings overflow'

															// 	continue
															// }

															// await env[`${vectorRegion}-${itemType}`].upsert($VectorizeVector)
														}

														console.log('JSON.stringify(tracking)',JSON.stringify(tracking));

														var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify(tracking.data)), { to: 'arraybuffer' })

														statements[`${zoneRegion}_tracking`].push(
															env[`${zoneRegion}_tracking`].prepare(`
																INSERT INTO tracking (
																	"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "index", "event", "goods", "order", "status", "no", "sender_address", "sender_phone", "recipient_address", "recipient_phone", "width", "height", "length", "weight", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "shipping_date", "delivery_date", "order_date", "payment_date", "payment_method", "payment_origin", "payment_number", "bundle_shipping"
																) VALUES (
																	?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31, ?32, ?33, ?34, ?35
																) ON CONFLICT (id) DO UPDATE SET
																	"type" = EXCLUDED."type",
																	"from" = EXCLUDED."from",
																	"to" = EXCLUDED."to",
																	"cc" = EXCLUDED."cc",
																	"bcc" = EXCLUDED."bcc",
																	"ref" = EXCLUDED."ref",
																	"data" = EXCLUDED."data",
																	"created_at" = EXCLUDED."created_at",
																	"index" = EXCLUDED."index",
																	"event" = EXCLUDED."event", 
																	"goods" = EXCLUDED."goods", 
																	"order" = EXCLUDED."order", 
																	"status" = EXCLUDED."status",
																	"no" = EXCLUDED."no",
																	"sender_address" = EXCLUDED."sender_address",
																	"sender_phone" = EXCLUDED."sender_phone",
																	"recipient_address" = EXCLUDED."recipient_address",
																	"recipient_phone" = EXCLUDED."recipient_phone",
																	"width" = EXCLUDED."width",
																	"height" = EXCLUDED."height",
																	"length" = EXCLUDED."length",
																	"weight" = EXCLUDED."weight",
																	"carrier" = EXCLUDED."carrier",
																	"shipping_fee" = EXCLUDED."shipping_fee",
																	"shipping_method" = EXCLUDED."shipping_method",
																	"shipping_duration" = EXCLUDED."shipping_duration",
																	"shipping_date" = EXCLUDED."shipping_date",
																	"delivery_date" = EXCLUDED."delivery_date",
																	"order_date" = EXCLUDED."order_date",
																	"payment_date" = EXCLUDED."payment_date",
																	"payment_method" = EXCLUDED."payment_method",
																	"payment_origin" = EXCLUDED."payment_origin",
																	"payment_number" = EXCLUDED."payment_number",
																	"bundle_shipping" = EXCLUDED."bundle_shipping"
															`).bind(
																tracking.id,
																tracking.type,
																tracking.from,
																tracking.to,
																tracking.cc,
																tracking.bcc,
																tracking.ref,
																arr.buffer,
																tracking.created_at,
																tracking.index,
																tracking.event ? tracking.event : 0,
																tracking.goods ? tracking.goods : 0,
																tracking.order ? tracking.order : 0,
																parseStatus(tracking.status),
																tracking.no ? tracking.no : "",
																tracking.sender_address ? tracking.sender_address : "",
																tracking.sender_phone ? tracking.sender_phone : "",
																tracking.recipient_address ? tracking.recipient_address : "",
																tracking.recipient_phone ? tracking.recipient_phone : "",
																parseFloat(tracking.width ? tracking.width : 0),
																parseFloat(tracking.height ? tracking.height : 0),
																parseFloat(tracking.length ? tracking.length : 0),
																parseFloat(tracking.weight ? tracking.weight : 0),
																parseFloat(tracking.carrier ? tracking.carrier : 0),
																parseFloat(tracking.shipping_fee ? tracking.shipping_fee : 0),
																tracking.shipping_method ? tracking.shipping_method : "",
																parseFloat(tracking.shipping_duration ? tracking.shipping_duration : 0),
																parseFloat(tracking.shipping_date ? tracking.shipping_date : 0),
																parseFloat(tracking.delivery_date ? tracking.delivery_date : 0),
																parseFloat(tracking.order_date ? tracking.order_date : 0),
																parseFloat(tracking.payment_date ? tracking.payment_date : 0),
																tracking.payment_method ? tracking.payment_method : "",
																tracking.payment_origin ? tracking.payment_origin : "",
																tracking.payment_number ? tracking.payment_number : "",
																parseFloat(tracking.bundle_shipping ? tracking.bundle_shipping : 0)
															)
														)
													}
												}
											}
										}catch(err){
											console.log('err item.type == "order" && item.tracking_number', err)
										}


										if(item.condition){
											if(item.condition.indexOf('used') > -1){
												item.used = 1
											}

											if(item.condition.indexOf('lease') > -1){
												item.lease = 1
											}

											if(item.condition.indexOf('rental') > -1){
												item.rental = 1
											}

											if(item.condition.indexOf('refurbish') > -1){
												item.refurbish = 1
											}
										}


										// team base 설정
										if(!team.data.base){
											team.data.base = {
												sales : {},
												event : {},
												tracking : {}
											}
										}



										var { results } = await env[`${zoneRegion}_${itemType}`].prepare(`SELECT * FROM ${itemType} WHERE "id" = "${item.id}" AND "to" = "${task.to}" AND "created_at" < ${now} LIMIT 1`).all()

										if(results.length){
											var _item = results[0]

											item = mergeNode(item, _item)
										}else{
											var { results } = await env[`${zoneRegion}_${itemType}`].prepare(`SELECT * FROM ${itemType} WHERE "index" = "${item.index}" AND "to" = "${task.to}" AND "created_at" < ${now} LIMIT 1`).all()

											if(results.length){
												var _item = results[0]

												item = mergeNode(item, _item)
											}
										}


										try{
											if(item.price <= team.data.base[item.type].price.min){
												team.data.base[item.type].price.min = item.price
											}

											if(item.price >= team.data.base[item.type].price.max){
												team.data.base[item.type].price.max = item.price
											}
											


											if(item.quantity <= team.data.base[item.type].quantity.min){
												team.data.base[item.type].quantity.min = item.quantity
											}

											if(item.quantity >= team.data.base[item.type].quantity.max){
												team.data.base[item.type].quantity.max = item.quantity
											}



											if(item.width <= team.data.base[item.type].width.min){
												team.data.base[item.type].width.min = item.width
											}

											if(item.width >= team.data.base[item.type].width.max){
												team.data.base[item.type].width.max = item.width
											}



											if(item.height <= team.data.base[item.type].height.min){
												team.data.base[item.type].height.min = item.height
											}

											if(item.height >= team.data.base[item.type].height.max){
												team.data.base[item.type].height.max = item.height
											}



											if(item.length <= team.data.base[item.type].length.min){
												team.data.base[item.type].length.min = item.length
											}

											if(item.length >= team.data.base[item.type].length.max){
												team.data.base[item.type].length.max = item.length
											}



											if(item.weight <= team.data.base[item.type].weight.min){
												team.data.base[item.type].weight.min = item.weight
											}

											if(item.weight >= team.data.base[item.type].weight.max){
												team.data.base[item.type].weight.max = item.weight
											}



											if(item.shipping_fee <= team.data.base[item.type].shipping_fee.min){
												team.data.base[item.type].shipping_fee.min = item.shipping_fee
											}

											if(item.shipping_fee >= team.data.base[item.type].shipping_fee.max){
												team.data.base[item.type].shipping_fee.max = item.shipping_fee
											}



											if(item.shipping_duration <= team.data.base[item.type].shipping_duration.min){
												team.data.base[item.type].shipping_duration.min = item.shipping_duration
											}

											if(item.shipping_duration >= team.data.base[item.type].shipping_duration.max){
												team.data.base[item.type].shipping_duration.max = item.shipping_duration
											}



											if(item.sale_price <= team.data.base[item.type].sale_price.min){
												team.data.base[item.type].sale_price.min = item.sale_price
											}

											if(item.sale_price >= team.data.base[item.type].sale_price.max){
												team.data.base[item.type].sale_price.max = item.sale_price
											}



											if(item.supply_price <= team.data.base[item.type].supply_price.min){
												team.data.base[item.type].supply_price.min = item.supply_price
											}

											if(item.supply_price >= team.data.base[item.type].supply_price.max){
												team.data.base[item.type].supply_price.max = item.supply_price
											}
											


											if(item.low_stock_threshold <= team.data.base[item.type].low_stock_threshold.min){
												team.data.base[item.type].low_stock_threshold.min = item.low_stock_threshold
											}

											if(item.low_stock_threshold >= team.data.base[item.type].low_stock_threshold.max){
												team.data.base[item.type].low_stock_threshold.max = item.low_stock_threshold
											}
											


											if(item.discount <= team.data.base[item.type].discount.min){
												team.data.base[item.type].discount.min = item.discount
											}

											if(item.discount >= team.data.base[item.type].discount.max){
												team.data.base[item.type].discount.max = item.discount
											}
											

											
											if(item.min_order_amount <= team.data.base[item.type].min_order_amount.min){
												team.data.base[item.type].min_order_amount.min = item.min_order_amount
											}

											if(item.min_order_amount >= team.data.base[item.type].min_order_amount.max){
												team.data.base[item.type].min_order_amount.max = item.min_order_amount
											}



											if(item.max_discount_amount <= team.data.base[item.type].max_discount_amount.min){
												team.data.base[item.type].max_discount_amount.min = item.max_discount_amount
											}

											if(item.max_discount_amount >= team.data.base[item.type].max_discount_amount.max){
												team.data.base[item.type].max_discount_amount.max = item.max_discount_amount
											}



											if(item.usage_limit <= team.data.base[item.type].usage_limit.min){
												team.data.base[item.type].usage_limit.min = item.usage_limit
											}

											if(item.usage_limit >= team.data.base[item.type].usage_limit.max){
												team.data.base[item.type].usage_limit.max = item.usage_limit
											}



											if(item.usage_per <= team.data.base[item.type].usage_per.min){
												team.data.base[item.type].usage_per.min = item.usage_per
											}

											if(item.usage_per >= team.data.base[item.type].usage_per.max){
												team.data.base[item.type].usage_per.max = item.usage_per
											}



											if(item.started_at <= team.data.base[item.type].started_at.min){
												team.data.base[item.type].started_at.min = item.started_at
											}

											if(item.started_at >= team.data.base[item.type].started_at.max){
												team.data.base[item.type].started_at.max = item.started_at
											}



											if(item.expired_at <= team.data.base[item.type].expired_at.min){
												team.data.base[item.type].expired_at.min = item.expired_at
											}

											if(item.expired_at >= team.data.base[item.type].expired_at.max){
												team.data.base[item.type].expired_at.max = item.expired_at
											}
										}catch(err){
											console.log('err team.data.base',err);
										}


										var updated_at

										var progress = {}

										var relates = {}

										var related = Related(item.type) // 관련 타입 정보 가져옴

										/*
											두가지 타입
												import
													foreign 에서 primary 

												export
													primary 에서 foreign 

												from : foreign 
												to : primary 
													import = 외부 데이터로 내부 데이터 수정
														order 스캔 진행시
															draft.type == "goods" && row.type == "order"
															
															order items 만 있으면 goods 상세 정보가 없기 때문에 
															goods 정보 가져와서 order item에 업데이트 해야함

												from : primary 
												to : foreign
													export = 내부 데이터로 외부 데이터 수정
														tracking 스캔 진행시
															tracking 정보는 있고, order 정보에 tracking 값 업데이트 해야함
										*/ 

										for(var r = 0; r < related.length; r++){
											var { query, merge } = Relay(related[r], item)

											// flow ${type}에 ${column} foreign 값이 없으면 업데이트 해야함

											if(!query || !merge){
												continue
											}

											try{
												if(query.length){
													var table = query[0].type 
													var type = query[0].type
													var column = query[0].column
													var column_value = query[0].value
													var status = query[0].status

													if(typeof status != "undefined"){
														var { results } = await env[`${zoneRegion}_${table}`].prepare(
															`SELECT * FROM ${table} WHERE "type" = "${type}" AND "${column}" = ? AND "to" = ? AND "cc" = ? AND "status" < ? AND "created_at" < ? ORDER BY created_at DESC LIMIT 1`
														).bind(
															column_value, team.id, item.cc, status, now
														).all()
													}else{
														var { results } = await env[`${zoneRegion}_${table}`].prepare(
															`SELECT * FROM ${table} WHERE "type" = "${type}" AND "${column}" = ? AND "to" = ? AND "cc" = ? AND "created_at" < ? ORDER BY created_at DESC LIMIT 1`
														).bind(
															column_value, team.id, item.cc, now
														).all()
													}

														

													if(results.length == 0){
														// draft 상태 맞음
														// 없으면 추가해야함 - 일부 사용자가 직접 팝업으로 띄워야 할수 있음

														/*
															상품 스캔 하였는데
															상품 상세페이지 스캔 안되어있으면
														*/

														/*
															고객 주문 스캔하였는데
															배송 시작 정보가 없을시
														*/
														updated_at = 0	
													}

													relates[type] = {
														query : query,
														merge : merge,
														rows : results,
														type : related[r]
													}
												}
											}catch(err){
												await env[CenterRegion].prepare(`
													INSERT INTO console (
														"id", "bcc", "log", "created_at"
													) VALUES (
														?1, ?2, ?3, ?4
													) ON CONFLICT (id) DO NOTHING
												`).bind(
													hashId(),
													task.bcc,
													'tracking inner err'+type+err,
													now // Parameter for created_at (only insert)
												).run()
											}
										}



										console.log('Object.keys(relates).length',Object.keys(relates).length);

										if(Object.keys(relates).length){
											for (var type in relates) {
												// for start

												// type값은 related[i]
												if (relates.hasOwnProperty(type)) {
													var relate = relates[type]

													/*
														시나리오 case

														import
															order 스캔 진행시
																type == "goods" && row.type == "order"

																order items 만 있으면 goods 상세 정보가 없기 때문에 
																goods 정보 가져와서 order item에 업데이트 해야함

														export
															tracking 스캔 진행시
																tracking 정보는 있고, order 정보에 tracking 값 업데이트 해야함

													*/
													
													var query = relate.query

													if(relate){
														if(typeof relate.rows != "undefined"){
															var column = query[1] ? query[1].column : query[0].column
															var index = query[1] ? query[1].index : query[0].index


															if(relate.rows.length){
																for(var d = 0; d < relate.rows.length; d++){
																	var row = relate.rows[d]

																	var nodeData = row.data

																	var merge

																	if(relate.merge.update){
																		merge = merge.update
																	}

																	if(relate.merge.upsert){
																		merge = merge.upsert
																	}

																	if(!merge){
																		continue
																	}

																	var foreign = merge.foreign

																	var node = {}

																	var from = row.type == merge.from ? item : row

																	var to = row.type == merge.to ? item : row
																	
																	if(merge.includes){
																		if(merge.includes.length){
																			for(var v = 0; v < merge.includes.length; v++){
																				var include = merge.includes[v]

																				node[include] = from[include]
																			}
																		}
																	}

																	


																	var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(from.data))

																	var data = JSON.parse(decompressedJsonString)

																	

																	if(foreign){
																		if(foreign.from && foreign.to){
																			if(from[foreign.to]){
																				to[foreign.from] = from[foreign.to]
																			}
																		}
																	}


																	var edgeId = to.id

																	var edgeType = to.type

																	var edge = mergeNode(to, node)

																	edge.id = edgeId

																	edge.type = edgeType

																	
																	if(relate.type == merge.from){
																		// import

																		if(from.type == "goods" && to.type == "order"){
																			

																			var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
																				id : edge.id,
																				title : edge.title,
																				link : edge.link,
																				origin : data.origin ? data.origin : "",
																				data : data
																			})), { to: 'arraybuffer' })

																			edge.vectorize = true

																			edge.data = arr.buffer

																			var metadata = {
																				title : data.title,
																				size : row.size ? row.size : "",
																				currency : row.currency ? row.currency : "",
																				carrier : row.carrier ? row.carrier : "",
																				shipping_fee : row.shipping_fee ? true : false,
																				shipping_method : row.shipping_method ? row.shipping_method : "",
																				fulfillment_service : row.fulfillment_service ? row.fulfillment_service : "",
																				stock_keeping_unit : row.stock_keeping_unit ? row.stock_keeping_unit : "",
																				bundle_shipping : row.bundle_shipping ? true : false,
																				used : row.used ? true : false,
																				lease : row.lease ? true : false,
																				rental : row.rental ? true : false,
																				refurbish : row.refurbish ? true : false,
																				tax_included : row.tax_included ? true : false
																			}


																			var content = JSON.stringify(metadata)

																			var semantic

																			if(models['deepinfra']){
																				semantic = await Deepinfra(deepinfra, 'openai/gpt-oss-20b', semantic_prompt_system(language), content)

																				models['deepinfra'] -= 1

																			}

																			if(!semantic & gemini_llm_api){
																				semantic = await Gemini(gemini_llm_api, gemini_llm_model, semantic_prompt_system(language), content, {"temperature": 1})

																				models[gemini_llm_api+'-'+gemini_llm_model] -= 1

																			}

																			if(!semantic){
																				fallback = 'semantic overflow'

																				continue
																			}

																			metadata.id = edge.id
																			metadata.type = edge.type
																			metadata.from = task.from
																			metadata.to = task.to
																			metadata.cc = task.cc
																			metadata.bcc = task.bcc
																			metadata.ref = task.ref

																			var embeddings

																			var $VectorizeVector

																			if(models['cloudflare']){
																				var { data: embeddings } = await env.AI.run('@cf/baai/bge-m3', {
																					text: [semantic],
																				})

																				var $VectorizeVector = [
																					{
																						id: edge.id,
																						values: embeddings[0],
																						metadata: metadata
																					}
																				]

																				models['cloudflare'] -= 1

																			}

																			if(!embeddings && models['deepinfra']){
																				var embeddings = await Deepinfra(deepinfra, 'BAAI/bge-m3', '', semantic)

																				var $VectorizeVector: VectorizeVector[] = embeddings.map((values, i) => {
																					return {
																						id: edge.id,
																						values: values,
																						metadata: metadata
																					}
																				})

																				models['deepinfra'] -= 1

																			}

																			if(embeddings){
																				fallback = 'embeddings overflow'

																				continue
																			}

																			await env[`${vectorRegion}-${type}`].upsert($VectorizeVector)

																		}
																	}else{
																		// export
																	}



																	if(edgeType == "sales"){
																		statements[`${zoneRegion}_sales`].push(
																			env[`${zoneRegion}_sales`].prepare(`
																				INSERT INTO sales (
																					"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "started_at", "expired_at", "index", "event", "views", "goods", "status", "width", "height", "length", "weight", "size", "currency", "supply_price", "sale_price", "discount", "quantity", "tracking", "number", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "fulfillment_service", "stock_keeping_unit", "bundle_shipping", "used", "lease", "rental", "refurbish", "tax_included", "release_date"
																				) VALUES (
																					?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31, ?32, ?33, ?34, ?35, ?36, ?37, ?38, ?39, ?40, ?41
																				) ON CONFLICT (id) DO UPDATE SET
																					"type" = EXCLUDED."type",
																					"from" = EXCLUDED."from",
																					"to" = EXCLUDED."to",
																					"cc" = EXCLUDED."cc",
																					"bcc" = EXCLUDED."bcc",
																					"ref" = EXCLUDED."ref",
																					"data" = EXCLUDED."data",
																					"created_at" = EXCLUDED."created_at",
																					"started_at" = EXCLUDED."started_at",
																					"expired_at" = EXCLUDED."expired_at",
																					"index" = EXCLUDED."index",
																					"event" = EXCLUDED."event",
																					"views" = EXCLUDED."views",
																					"goods" = EXCLUDED."goods",
																					"status" = EXCLUDED."status",
																					"width" = EXCLUDED."width",
																					"height" = EXCLUDED."height",
																					"length" = EXCLUDED."length",
																					"weight" = EXCLUDED."weight",
																					"size" = EXCLUDED."size",
																					"currency" = EXCLUDED."currency",
																					"supply_price" = EXCLUDED."supply_price",
																					"sale_price" = EXCLUDED."sale_price",
																					"discount" = EXCLUDED."discount",
																					"quantity" = EXCLUDED."quantity",
																					"tracking" = EXCLUDED."tracking",
																					"number" = EXCLUDED."number",
																					"carrier" = EXCLUDED."carrier",
																					"shipping_fee" = EXCLUDED."shipping_fee",
																					"shipping_method" = EXCLUDED."shipping_method",
																					"shipping_duration" = EXCLUDED."shipping_duration",
																					"fulfillment_service" = EXCLUDED."fulfillment_service",
																					"stock_keeping_unit" = EXCLUDED."stock_keeping_unit",
																					"bundle_shipping" = EXCLUDED."bundle_shipping",
																					"used" = EXCLUDED."used",
																					"lease" = EXCLUDED."lease",
																					"rental" = EXCLUDED."rental",
																					"refurbish" = EXCLUDED."refurbish",
																					"tax_included" = EXCLUDED."tax_included",
																					"release_date" = EXCLUDED."release_date"
																			`).bind(
																				to.id,
																				edgeType,
																				to.from,
																				to.to,
																				to.cc,
																				to.bcc,
																				to.ref,
																				to.data,
																				to.created_at,
																				parseFloat(edge.started_at ? edge.started_at : 0),
																				parseFloat(edge.expired_at ? edge.expired_at : 0),
																				parseFloat(edge.index ? edge.index : 0),
																				parseFloat(edge.event ? edge.event : 0),
																				parseFloat(edge.views ? edge.views : 0),
																				parseFloat(edge.goods ? edge.goods : 0),
																				parseStatus(edge.status),
																				parseFloat(edge.width ? edge.width : 0),
																				parseFloat(edge.height ? edge.height : 0),
																				parseFloat(edge.length ? edge.length : 0),
																				parseFloat(edge.weight ? edge.weight : 0),
																				edge.size ? edge.size : "",
																				edge.currency,
																				parseFloat(edge.supply_price? edge.supply_price : 0),
																				parseFloat(edge.sale_price? edge.sale_price : 0),
																				parseFloat(edge.discount ? edge.discount : 0),
																				parseFloat(edge.quantity ? edge.quantity : 0),
																				parseFloat(edge.tracking ? edge.tracking : 0),
																				edge.number ? edge.number : "",
																				edge.carrier ? edge.carrier : "",
																				parseFloat(edge.shipping_fee ? edge.shipping_fee : 0),
																				edge.shipping_method ? edge.shipping_method : "",
																				parseFloat(edge.shipping_duration ? edge.shipping_duration : 0),
																				edge.fulfillment_service ? edge.fulfillment_service : "",
																				edge.stock_keeping_unit ? edge.stock_keeping_unit : "",
																				parseFloat(edge.bundle_shipping ? edge.bundle_shipping : 0),
																				parseFloat(edge.used ? edge.used : 0),
																				parseFloat(edge.lease ? edge.lease : 0),
																				parseFloat(edge.rental ? edge.rental : 0),
																				parseFloat(edge.refurbish ? edge.refurbish : 0),
																				parseFloat(edge.tax_included ? edge.tax_included : 0),
																				parseFloat(edge.release_date ? edge.release_date : 0)
																			)
																		)
																	}else if(edgeType == "tracking"){
																		statements[`${zoneRegion}_tracking`].push(
																			env[`${zoneRegion}_tracking`].prepare(`
																				INSERT INTO tracking (
																					"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "index", "event", "goods", "order", "status", "no", "sender_address", "sender_phone", "recipient_address", "recipient_phone", "width", "height", "length", "weight", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "shipping_date", "delivery_date", "order_date", "payment_date", "payment_method", "payment_origin", "payment_number", "bundle_shipping"
																				) VALUES (
																					?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31, ?32, ?33, ?34, ?35
																				) ON CONFLICT (id) DO UPDATE SET
																					"type" = EXCLUDED."type",
																					"from" = EXCLUDED."from",
																					"to" = EXCLUDED."to",
																					"cc" = EXCLUDED."cc",
																					"bcc" = EXCLUDED."bcc",
																					"ref" = EXCLUDED."ref",
																					"data" = EXCLUDED."data",
																					"created_at" = EXCLUDED."created_at",
																					"index" = EXCLUDED."index",
																					"event" = EXCLUDED."event", 
																					"goods" = EXCLUDED."goods", 
																					"order" = EXCLUDED."order", 
																					"status" = EXCLUDED."status",
																					"no" = EXCLUDED."no",
																					"sender_address" = EXCLUDED."sender_address",
																					"sender_phone" = EXCLUDED."sender_phone",
																					"recipient_address" = EXCLUDED."recipient_address",
																					"recipient_phone" = EXCLUDED."recipient_phone",
																					"width" = EXCLUDED."width",
																					"height" = EXCLUDED."height",
																					"length" = EXCLUDED."length",
																					"weight" = EXCLUDED."weight",
																					"carrier" = EXCLUDED."carrier",
																					"shipping_fee" = EXCLUDED."shipping_fee",
																					"shipping_method" = EXCLUDED."shipping_method",
																					"shipping_duration" = EXCLUDED."shipping_duration",
																					"shipping_date" = EXCLUDED."shipping_date",
																					"delivery_date" = EXCLUDED."delivery_date",
																					"order_date" = EXCLUDED."order_date",
																					"payment_date" = EXCLUDED."payment_date",
																					"payment_method" = EXCLUDED."payment_method",
																					"payment_origin" = EXCLUDED."payment_origin",
																					"payment_number" = EXCLUDED."payment_number",
																					"bundle_shipping" = EXCLUDED."bundle_shipping"
																			`).bind(
																				to.id,
																				to.type,
																				to.from,
																				to.to,
																				to.cc,
																				to.bcc,
																				to.ref,
																				to.data,
																				to.created_at,
																				edge.index,
																				edge.event ? edge.event : 0,
																				edge.goods ? edge.goods : 0,
																				edge.order ? edge.order : 0,
																				parseStatus(edge.status),
																				edge.no ? edge.no : "",
																				edge.sender_address ? edge.sender_address : "",
																				edge.sender_phone ? edge.sender_phone : "",
																				edge.recipient_address ? edge.recipient_address : "",
																				edge.recipient_phone ? edge.recipient_phone : "",
																				parseFloat(edge.width ? edge.width : 0),
																				parseFloat(edge.height ? edge.height : 0),
																				parseFloat(edge.length ? edge.length : 0),
																				parseFloat(edge.weight ? edge.weight : 0),
																				parseFloat(edge.carrier ? edge.carrier : 0),
																				parseFloat(edge.shipping_fee ? edge.shipping_fee : 0),
																				edge.shipping_method ? edge.shipping_method : "",
																				parseFloat(edge.shipping_duration ? edge.shipping_duration : 0),
																				parseFloat(edge.shipping_date ? edge.shipping_date : 0),
																				parseFloat(edge.delivery_date ? edge.delivery_date : 0),
																				parseFloat(edge.order_date ? edge.order_date : 0),
																				parseFloat(edge.payment_date ? edge.payment_date : 0),
																				edge.payment_method ? edge.payment_method : "",
																				edge.payment_origin ? edge.payment_origin : "",
																				edge.payment_number ? edge.payment_number : "",
																				parseFloat(edge.bundle_shipping ? edge.bundle_shipping : 0)
																			)
																		)
																	}else if(edgeType == "event"){
																		statements[`${zoneRegion}_event`].push(
																			env[`${zoneRegion}_event`].prepare(`
																				INSERT INTO event (
																					"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "started_at", "expired_at", "index", "event", "number", "address", "status", "code", "discount", "quantity", "usage_per", "usage_limit", "min_order_amount", "max_order_amount", "max_discount_amount", "new_customer_only", "first_purchase_only", "region_restrictions"
																				) VALUES (
																					?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27
																				) ON CONFLICT (id) DO UPDATE SET
																					"type" = EXCLUDED."type",
																					"from" = EXCLUDED."from",
																					"to" = EXCLUDED."to",
																					"cc" = EXCLUDED."cc",
																					"bcc" = EXCLUDED."bcc",
																					"ref" = EXCLUDED."ref",
																					"data" = EXCLUDED."data",
																					"created_at" = EXCLUDED."created_at",
																					"started_at" = EXCLUDED."started_at",
																					"expired_at" = EXCLUDED."expired_at",
																					"index" = EXCLUDED."index",
																					"event" = EXCLUDED."event",
																					"number" = EXCLUDED."number",
																					"address" = EXCLUDED."address",
																					"status" = EXCLUDED."status",
																					"code" = EXCLUDED."code",
																					"discount" = EXCLUDED."discount",
																					"quantity" = EXCLUDED."quantity",
																					"usage_per" = EXCLUDED."usage_per",
																					"usage_limit" = EXCLUDED."usage_limit",
																					"min_order_amount" = EXCLUDED."min_order_amount",
																					"max_order_amount" = EXCLUDED."max_order_amount",
																					"max_discount_amount" = EXCLUDED."max_discount_amount",
																					"new_customer_only" = EXCLUDED."new_customer_only",
																					"first_purchase_only" = EXCLUDED."first_purchase_only",
																					"region_restrictions" = EXCLUDED."region_restrictions"
																			`).bind(
																				to.id,
																				edgeType,
																				to.from,
																				to.to,
																				to.cc,
																				to.bcc,
																				to.ref,
																				to.data,
																				to.created_at,
																				parseFloat(edge.started_at ? edge.started_at : 0),
																				parseFloat(edge.expired_at ? edge.expired_at : 0),
																				parseFloat(edge.index ? edge.index : 0),
																				parseFloat(edge.event ? edge.event : 0),
																				edge.number ? edge.number : "",
																				edge.address ? edge.address : "",
																				parseStatus(edge.status),
																				edge.code ? edge.code : "",
																				parseFloat(edge.discount ? edge.discount : 0),
																				parseFloat(edge.quantity ? edge.quantity : 0),
																				parseFloat(edge.usage_per ? edge.usage_per : 0),
																				parseFloat(edge.usage_limit ? edge.usage_limit : 0),
																				parseFloat(edge.min_order_amount ? edge.min_order_amount : 0),
																				parseFloat(edge.max_order_amount ? edge.max_order_amount : 0),
																				parseFloat(edge.max_discount_amount ? edge.max_discount_amount : 0),
																				parseFloat(edge.new_customer_only ? edge.new_customer_only : 0),
																				parseFloat(edge.first_purchase_only ? edge.first_purchase_only : 0),
																				parseFloat(edge.region_restrictions ? edge.region_restrictions : 0)
																			)
																		)
																	}



																	// before ${type}에 ${column} index 값이 없으면 업데이트 해야함
																	statements[`${zoneRegion}_items`].push(
																		env[`${zoneRegion}_items`].prepare(`
																			UPDATE items SET updated_at = ? WHERE id = ?
																		`).bind(
																			now, row.id
																		)
																	)

																	// for end
																}

																// if end
															}else{
																// draft 추가해야함
																var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
																	id : item.id,
																	title : item.title,
																	link : item.link,
																	data : relate
																})), { to: 'arraybuffer' })

																statements[`${zoneRegion}_items`].push(
																	env[`${zoneRegion}_items`].prepare(`
																		INSERT INTO items (
																			"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "updated_at"
																		) VALUES (
																			?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10
																		) ON CONFLICT (id) DO UPDATE SET
																			"type" = EXCLUDED."type",
																			"from" = EXCLUDED."from",
																			"to" = EXCLUDED."to",
																			"cc" = EXCLUDED."cc",
																			"bcc" = EXCLUDED."bcc",
																			"ref" = EXCLUDED."ref",
																			"data" = EXCLUDED."data",
																			"created_at" = EXCLUDED."created_at",
																			"updated_at" = EXCLUDED."updated_at"
																	`).bind(
																		hashId(),
																		type,
																		item.from,
																		item.to,
																		item.cc,
																		item.bcc,
																		'',
																		arr.buffer,
																		now,
																		0
																	)
																)
															}
														}
													}
												}

												// for end
											}

											// if end
										}

										if(item.semantic && !item.vectorize){
											var metadata = {
												type: item.type,
												from: item.from,
												to: item.to,
												cc: item.cc,
												bcc: item.bcc,
												ref:task.ref
											}

											var embeddings

											if(models['cloudflare']){
												var { data: embeddings } = await env.AI.run('@cf/baai/bge-m3', {
													text: [item.semantic]
												})

												var $VectorizeVector = [
													{
														id: item.id,
														values: embeddings[0],
														metadata: metadata
													}
												]

												models['cloudflare'] -= 1

											}

											if(!embeddings && models['deepinfra']){
												var embeddings = await Deepinfra(deepinfra, 'BAAI/bge-m3', '', item.semantic)

												var $VectorizeVector: VectorizeVector[] = embeddings.map((values, i) => {
													return {
														id: item.id,
														values: values,
														metadata: metadata
													}
												})

												models['deepinfra'] -= 1
											}

											console.log('typeof embeddings',typeof embeddings);

											if(!embeddings){
												fallback = 'embeddings overflow'

												continue
											}

											await env[`${vectorRegion}-${itemType}`].upsert($VectorizeVector)
										}


										var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
											type : item.type,
											text : item.semantic,
											link : item.link,
											origin : task.origin ? task.origin : ''
										})), { to: 'arraybuffer' })


										statements[`${zoneRegion}_items`].push(
											env[`${zoneRegion}_items`].prepare(`
												INSERT INTO items (
													"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "updated_at"
												) VALUES (
													?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10
												) ON CONFLICT (id) DO UPDATE SET
													"type" = EXCLUDED."type",
													"from" = EXCLUDED."from",
													"to" = EXCLUDED."to",
													"cc" = EXCLUDED."cc",
													"bcc" = EXCLUDED."bcc",
													"ref" = EXCLUDED."ref",
													"data" = EXCLUDED."data",
													"created_at" = EXCLUDED."created_at",
													"updated_at" = EXCLUDED."updated_at"
											`).bind(
												item.id,
												item.type,
												item.from,
												item.to,
												item.cc,
												item.bcc,
												item.ref,
												arr.buffer,
												now,
												typeof updated_at != "undefined" ? updated_at : now 
											)
										)

										if(itemType == "sales"){
											statements[`${zoneRegion}_sales`].push(
												env[`${zoneRegion}_sales`].prepare(`
													INSERT INTO sales (
														"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "started_at", "expired_at", "index", "event", "views", "goods", "status", "width", "height", "length", "weight", "size", "currency", "supply_price", "sale_price", "discount", "quantity", "tracking", "number", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "fulfillment_service", "stock_keeping_unit", "bundle_shipping", "used", "lease", "rental", "refurbish", "tax_included", "release_date"
													) VALUES (
														?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31, ?32, ?33, ?34, ?35, ?36, ?37, ?38, ?39, ?40, ?41
													) ON CONFLICT (id) DO UPDATE SET
														"type" = EXCLUDED."type",
														"from" = EXCLUDED."from",
														"to" = EXCLUDED."to",
														"cc" = EXCLUDED."cc",
														"bcc" = EXCLUDED."bcc",
														"ref" = EXCLUDED."ref",
														"data" = EXCLUDED."data",
														"created_at" = EXCLUDED."created_at",
														"started_at" = EXCLUDED."started_at",
														"expired_at" = EXCLUDED."expired_at",
														"index" = EXCLUDED."index",
														"event" = EXCLUDED."event",
														"views" = EXCLUDED."views",
														"goods" = EXCLUDED."goods",
														"status" = EXCLUDED."status",
														"width" = EXCLUDED."width",
														"height" = EXCLUDED."height",
														"length" = EXCLUDED."length",
														"weight" = EXCLUDED."weight",
														"size" = EXCLUDED."size",
														"currency" = EXCLUDED."currency",
														"supply_price" = EXCLUDED."supply_price",
														"sale_price" = EXCLUDED."sale_price",
														"discount" = EXCLUDED."discount",
														"quantity" = EXCLUDED."quantity",
														"tracking" = EXCLUDED."tracking",
														"number" = EXCLUDED."number",
														"carrier" = EXCLUDED."carrier",
														"shipping_fee" = EXCLUDED."shipping_fee",
														"shipping_method" = EXCLUDED."shipping_method",
														"shipping_duration" = EXCLUDED."shipping_duration",
														"fulfillment_service" = EXCLUDED."fulfillment_service",
														"stock_keeping_unit" = EXCLUDED."stock_keeping_unit",
														"bundle_shipping" = EXCLUDED."bundle_shipping",
														"used" = EXCLUDED."used",
														"lease" = EXCLUDED."lease",
														"rental" = EXCLUDED."rental",
														"refurbish" = EXCLUDED."refurbish",
														"tax_included" = EXCLUDED."tax_included",
														"release_date" = EXCLUDED."release_date"
												`).bind(
													item.id,
													item.type,
													item.from,
													item.to,
													item.cc,
													item.bcc,
													item.ref,
													item.data,
													item.created_at,
													parseFloat(item.started_at ? item.started_at : 0),
													parseFloat(item.expired_at ? item.expired_at : 0),
													parseFloat(item.index ? item.index : 0),
													parseFloat(item.event ? item.event : 0),
													parseFloat(item.views ? item.views : 0),
													parseFloat(item.goods ? item.goods : 0),
													parseStatus(item.status),
													parseFloat(item.width ? item.width : 0),
													parseFloat(item.height ? item.height : 0),
													parseFloat(item.length ? item.length : 0),
													parseFloat(item.weight ? item.weight : 0),
													item.size ? item.size : "",
													item.currency,
													parseFloat(item.supply_price? item.supply_price : 0),
													parseFloat(item.sale_price? item.sale_price : 0),
													parseFloat(item.discount ? item.discount : 0),
													parseFloat(item.quantity ? item.quantity : 0),
													parseFloat(item.tracking ? item.tracking : 0),
													item.number ? item.number : "",
													item.carrier ? item.carrier : "",
													parseFloat(item.shipping_fee ? item.shipping_fee : 0),
													item.shipping_method ? item.shipping_method : "",
													parseFloat(item.shipping_duration ? item.shipping_duration : 0),
													item.fulfillment_service ? item.fulfillment_service : "",
													item.stock_keeping_unit ? item.stock_keeping_unit : "",
													parseFloat(item.bundle_shipping ? item.bundle_shipping : 0),
													parseFloat(item.used ? item.used : 0),
													parseFloat(item.lease ? item.lease : 0),
													parseFloat(item.rental ? item.rental : 0),
													parseFloat(item.refurbish ? item.refurbish : 0),
													parseFloat(item.tax_included ? item.tax_included : 0),
													parseFloat(item.release_date ? item.release_date : 0)
												)
											)
										}else if(itemType == "tracking"){
											statements[`${zoneRegion}_tracking`].push(
												env[`${zoneRegion}_tracking`].prepare(`
													INSERT INTO tracking (
														"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "index", "event", "goods", "order", "status", "no", "sender_address", "sender_phone", "recipient_address", "recipient_phone", "width", "height", "length", "weight", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "shipping_date", "delivery_date", "order_date", "payment_date", "payment_method", "payment_origin", "payment_number", "bundle_shipping"
													) VALUES (
														?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31, ?32, ?33, ?34, ?35
													) ON CONFLICT (id) DO UPDATE SET
														"type" = EXCLUDED."type",
														"from" = EXCLUDED."from",
														"to" = EXCLUDED."to",
														"cc" = EXCLUDED."cc",
														"bcc" = EXCLUDED."bcc",
														"ref" = EXCLUDED."ref",
														"data" = EXCLUDED."data",
														"created_at" = EXCLUDED."created_at",
														"index" = EXCLUDED."index",
														"event" = EXCLUDED."event", 
														"goods" = EXCLUDED."goods", 
														"order" = EXCLUDED."order", 
														"status" = EXCLUDED."status",
														"no" = EXCLUDED."no",
														"sender_address" = EXCLUDED."sender_address",
														"sender_phone" = EXCLUDED."sender_phone",
														"recipient_address" = EXCLUDED."recipient_address",
														"recipient_phone" = EXCLUDED."recipient_phone",
														"width" = EXCLUDED."width",
														"height" = EXCLUDED."height",
														"length" = EXCLUDED."length",
														"weight" = EXCLUDED."weight",
														"carrier" = EXCLUDED."carrier",
														"shipping_fee" = EXCLUDED."shipping_fee",
														"shipping_method" = EXCLUDED."shipping_method",
														"shipping_duration" = EXCLUDED."shipping_duration",
														"shipping_date" = EXCLUDED."shipping_date",
														"delivery_date" = EXCLUDED."delivery_date",
														"order_date" = EXCLUDED."order_date",
														"payment_date" = EXCLUDED."payment_date",
														"payment_method" = EXCLUDED."payment_method",
														"payment_origin" = EXCLUDED."payment_origin",
														"payment_number" = EXCLUDED."payment_number",
														"bundle_shipping" = EXCLUDED."bundle_shipping"
												`).bind(
													item.id,
													item.type,
													item.from,
													item.to,
													item.cc,
													item.bcc,
													item.ref,
													item.data,
													item.created_at,
													item.index,
													item.event ? item.event : 0,
													item.goods ? item.goods : 0,
													item.order ? item.order : 0,
													parseStatus(item.status),
													item.no ? item.no : "",
													item.sender_address ? item.sender_address : "",
													item.sender_phone ? item.sender_phone : "",
													item.recipient_address ? item.recipient_address : "",
													item.recipient_phone ? item.recipient_phone : "",
													parseFloat(item.width ? item.width : 0),
													parseFloat(item.height ? item.height : 0),
													parseFloat(item.length ? item.length : 0),
													parseFloat(item.weight ? item.weight : 0),
													parseFloat(item.carrier ? item.carrier : 0),
													parseFloat(item.shipping_fee ? item.shipping_fee : 0),
													item.shipping_method ? item.shipping_method : "",
													parseFloat(item.shipping_duration ? item.shipping_duration : 0),
													parseFloat(item.shipping_date ? item.shipping_date : 0),
													parseFloat(item.delivery_date ? item.delivery_date : 0),
													parseFloat(item.order_date ? item.order_date : 0),
													parseFloat(item.payment_date ? item.payment_date : 0),
													item.payment_method ? item.payment_method : "",
													item.payment_origin ? item.payment_origin : "",
													item.payment_number ? item.payment_number : "",
													parseFloat(item.bundle_shipping ? item.bundle_shipping : 0)
												)
											)
										}else if(itemType == "event"){
											statements[`${zoneRegion}_event`].push(
												env[`${zoneRegion}_event`].prepare(`
													INSERT INTO event (
														"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "started_at", "expired_at", "index", "event", "number", "address", "status", "code", "discount", "quantity", "usage_per", "usage_limit", "min_order_amount", "max_order_amount", "max_discount_amount", "new_customer_only", "first_purchase_only", "region_restrictions"
													) VALUES (
														?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27
													) ON CONFLICT (id) DO UPDATE SET
														"type" = EXCLUDED."type",
														"from" = EXCLUDED."from",
														"to" = EXCLUDED."to",
														"cc" = EXCLUDED."cc",
														"bcc" = EXCLUDED."bcc",
														"ref" = EXCLUDED."ref",
														"data" = EXCLUDED."data",
														"created_at" = EXCLUDED."created_at",
														"started_at" = EXCLUDED."started_at",
														"expired_at" = EXCLUDED."expired_at",
														"index" = EXCLUDED."index",
														"event" = EXCLUDED."event",
														"number" = EXCLUDED."number",
														"address" = EXCLUDED."address",
														"status" = EXCLUDED."status",
														"code" = EXCLUDED."code",
														"discount" = EXCLUDED."discount",
														"quantity" = EXCLUDED."quantity",
														"usage_per" = EXCLUDED."usage_per",
														"usage_limit" = EXCLUDED."usage_limit",
														"min_order_amount" = EXCLUDED."min_order_amount",
														"max_order_amount" = EXCLUDED."max_order_amount",
														"max_discount_amount" = EXCLUDED."max_discount_amount",
														"new_customer_only" = EXCLUDED."new_customer_only",
														"first_purchase_only" = EXCLUDED."first_purchase_only",
														"region_restrictions" = EXCLUDED."region_restrictions"
												`).bind(
													item.id,
													item.type,
													item.from,
													item.to,
													item.cc,
													item.bcc,
													item.ref,
													item.data,
													item.created_at,
													parseFloat(item.started_at ? item.started_at : 0),
													parseFloat(item.expired_at ? item.expired_at : 0),
													parseFloat(item.index ? item.index : 0),
													parseFloat(item.event ? item.event : 0),
													item.number ? item.number : "",
													item.address ? item.address : "",
													parseStatus(item.status),
													item.code ? item.code : "",
													parseFloat(item.discount ? item.discount : 0),
													parseFloat(item.quantity ? item.quantity : 0),
													parseFloat(item.usage_per ? item.usage_per : 0),
													parseFloat(item.usage_limit ? item.usage_limit : 0),
													parseFloat(item.min_order_amount ? item.min_order_amount : 0),
													parseFloat(item.max_order_amount ? item.max_order_amount : 0),
													parseFloat(item.max_discount_amount ? item.max_discount_amount : 0),
													parseFloat(item.new_customer_only ? item.new_customer_only : 0),
													parseFloat(item.first_purchase_only ? item.first_purchase_only : 0),
													parseFloat(item.region_restrictions ? item.region_restrictions : 0)
												)
											)
										}
									}
								}



								task.title = page.text // Analyze the provided Pug template and return it in the following JSON format

								task.semantic = page.text

								if(items.length){
									var type = page.type

									if(page.type == "sales"){
										type = "sales"

									}else if(page.type == "goods" || page.type == "order"){
										type = "sales"

									}else if(page.type == "event" || page.type == "coupon"){
										type = "event"

									}

									talk.type = type


									var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
										no : task.no,
										title : page.text,
										semantic : task.semantic
									})), { to: 'arraybuffer' })

									task.data = arr.buffer
								}else{
									/*
										정보가 없으면

										클라이언트에서 정보를 찾지 못하였다고 안내하기

										가능한 카테고리 안내 메세지 노출해야함
									*/ 

									talk.type = "empty"

									task.data = null
								}

								var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
									type : page.type,
									text : task.semantic,
									link : task.link,
									origin : task.origin ? task.origin : ''
								})), { to: 'arraybuffer' })

								statements[`${zoneRegion}_items`].push(
									env[`${zoneRegion}_items`].prepare(`
										INSERT INTO items (
											"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "updated_at"
										) VALUES (
											?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10
										) ON CONFLICT (id) DO UPDATE SET
											"type" = EXCLUDED."type",
											"from" = EXCLUDED."from",
											"to" = EXCLUDED."to",
											"cc" = EXCLUDED."cc",
											"bcc" = EXCLUDED."bcc",
											"ref" = EXCLUDED."ref",
											"data" = EXCLUDED."data",
											"created_at" = EXCLUDED."created_at",
											"updated_at" = EXCLUDED."updated_at"
									`).bind(
										task.id,
										task.type,
										task.from,
										task.to,
										task.cc,
										task.bcc,
										task.ref,
										arr.buffer,
										now,
										now
									)
								)

							}catch(err){
								console.log('inner err '+err)

								await env[CenterRegion].prepare(`
									INSERT INTO console (
										"id", "bcc", "log", "created_at"
									) VALUES (
										?1, ?2, ?3, ?4
									) ON CONFLICT (id) DO NOTHING
								`).bind(
									hashId(),
									task.bcc,
									'inner err'+err,
									now // Parameter for created_at (only insert)
								).run()
							}

						}else{
							// SELECT 백터 쿼리

							// 2022년 이후 2023년에 A 쇼핑몰에서 가장 많이 팔린 제품은 뭐야?

							/*
								추후 few shot 추가하기
								
								1. 이전 대화 참조
									id 뎁스 여러번 hashId 거친것 select해서 있으면 이전 대화 참조하기

								2. 프리미엄 사용자
									vectorize에 프롬프트 semantic 결과값 추가하고 연관된 애들 불러와서 결과 만들기

									아직 개발 안됨


								프롬프트 답변값 벡터 데이터 저장 되어있음

								env[CenterRegion]에 저장 되어있어서 그쪽에서 쿼리해야함

							*/ 

							
							var paragraphs

							if(models['deepinfra']){
								var res = await Deepinfra(deepinfra, 'openai/gpt-oss-20b', para2graph(language).trim(), task.text)

								if(res){
									paragraphs = res.context
								}

								models['deepinfra'] -= 1

							}

							if(!paragraphs && gemini_llm_api){
								var res = await Gemini(gemini_llm_api, gemini_llm_model, para2graph(language).trim(), task.text)

								if(res){
									paragraphs = res.context
								}

								models[gemini_llm_api+'-'+gemini_llm_model] -= 1

							}

							if(!paragraphs){
								fallback = 'para2graph overflow'

								continue
							}


							try{
								if(paragraphs.length){
									for(var p = 0; p < paragraphs.length; p++){
										var paragraph = paragraphs[p]

										paragraph.status = null
										paragraph.orderBy = null
										paragraph.find = null

										if(!paragraph.price){
											paragraph.price = {}
										}


										var type = paragraph.type

										if(paragraph.type == "sales"){
											type = "sales"

											paragraph.type = "order"

										}else if(paragraph.type == "goods" || paragraph.type == "order"){
											type = "sales"

										}else if(paragraph.type == "event" || paragraph.type == "coupon"){
											type = "event"

										}

										if(team.data.base[paragraph.type]?.price.min){
											paragraph.price.min = `min:${team.data.base[paragraph.type]?.price.min},`
										}

										if(team.data.base[paragraph.type]?.price.max){
											paragraph.price.max = `max:${team.data.base[paragraph.type]?.price.max},`
										}

										

										if(!paragraph.quantity){
											paragraph.quantity = {}
										}

										if(team.data.base[paragraph.type]?.quantity.min){
											paragraph.quantity.min = `min:${team.data.base[paragraph.type]?.quantity.min},`
										}

										if(team.data.base[paragraph.type]?.quantity.max){
											paragraph.quantity.max = `max:${team.data.base[paragraph.type]?.quantity.max},`
										}



										if(!paragraph.width){
											paragraph.width = {}
										}

										if(team.data.base[paragraph.type]?.width.min){
											paragraph.width.min = `min:${team.data.base[paragraph.type]?.width.min},`
										}

										if(team.data.base[paragraph.type]?.width.max){
											paragraph.width.max = `max:${team.data.base[paragraph.type]?.width.max},`
										}



										if(!paragraph.height){
											paragraph.height = {}
										}

										if(team.data.base[paragraph.type]?.height.min){
											paragraph.height.min = `min:${team.data.base[paragraph.type]?.height.min},`
										}

										if(team.data.base[paragraph.type]?.height.max){
											paragraph.height.max = `max:${team.data.base[paragraph.type]?.height.max},`
										}



										if(!paragraph.length){
											paragraph.length = {}
										}

										if(team.data.base[paragraph.type]?.length.min){
											paragraph.length.min = `min:${team.data.base[paragraph.type]?.length.min},`
										}

										if(team.data.base[paragraph.type]?.length.max){
											paragraph.length.max = `max:${team.data.base[paragraph.type]?.length.max},`
										}



										if(!paragraph.weight){
											paragraph.weight = {}
										}

										if(team.data.base[paragraph.type]?.weight.min){
											paragraph.weight.min = `min:${team.data.base[paragraph.type]?.weight.min},`
										}

										if(team.data.base[paragraph.type]?.weight.max){
											paragraph.weight.max = `max:${team.data.base[paragraph.type]?.weight.max},`
										}



										if(!paragraph.shipping_fee){
											paragraph.shipping_fee = {}
										}

										if(team.data.base[paragraph.type]?.shipping_fee.min){
											paragraph.shipping_fee.min = `min:${team.data.base[paragraph.type]?.shipping_fee.min},`
										}

										if(team.data.base[paragraph.type]?.shipping_fee.max){
											paragraph.shipping_fee.max = `max:${team.data.base[paragraph.type]?.shipping_fee.max},`
										}



										if(!paragraph.shipping_duration){
											paragraph.shipping_duration = {}
										}

										if(team.data.base[paragraph.type]?.shipping_duration.min){
											paragraph.shipping_duration.min = `min:${team.data.base[paragraph.type]?.shipping_duration.min},`
										}

										if(team.data.base[paragraph.type]?.shipping_duration.max){
											paragraph.shipping_duration.max = `max:${team.data.base[paragraph.type]?.shipping_duration.max},`
										}



										if(!paragraph.price){
											paragraph.price = {}
										}

										if(team.data.base[paragraph.type]?.sale_price.min){
											paragraph.price.min = `min:${team.data.base[paragraph.type]?.sale_price.min},`
										}

										if(team.data.base[paragraph.type]?.sale_price.max){
											paragraph.price.max = `max:${team.data.base[paragraph.type]?.sale_price.max},`
										}



										if(!paragraph.supply_price){
											paragraph.supply_price = {}
										}

										if(team.data.base[paragraph.type]?.supply_price.min){
											paragraph.supply_price.min = `min:${team.data.base[paragraph.type]?.supply_price.min},`
										}
										
										if(team.data.base[paragraph.type]?.supply_price.max){
											paragraph.supply_price.max = `max:${team.data.base[paragraph.type]?.supply_price.max},`
										}




										if(!paragraph.low_stock_threshold){
											paragraph.low_stock_threshold = {}
										}

										if(team.data.base[paragraph.type]?.low_stock_threshold.min){
											paragraph.low_stock_threshold.min = `min:${team.data.base[paragraph.type]?.low_stock_threshold.min},`
										}
										
										if(team.data.base[paragraph.type]?.low_stock_threshold.max){
											paragraph.low_stock_threshold.max = `max:${team.data.base[paragraph.type]?.low_stock_threshold.max},`
										}



										if(!paragraph.discount){
											paragraph.discount = {}
										}

										if(team.data.base[paragraph.type]?.discount.min){
											paragraph.discount.min = `min:${team.data.base[paragraph.type]?.discount.min},`
										}

										if(team.data.base[paragraph.type]?.discount.max){
											paragraph.discount.max = `max:${team.data.base[paragraph.type]?.discount.max},`
										}



										if(!paragraph.min_order_amount){
											paragraph.min_order_amount = {}
										}

										if(team.data.base[paragraph.type]?.min_order_amount.min){
											paragraph.min_order_amount.min = `min:${team.data.base[paragraph.type]?.min_order_amount.min},`
										}

										if(team.data.base[paragraph.type]?.min_order_amount.max){
											paragraph.min_order_amount.max = `max:${team.data.base[paragraph.type]?.min_order_amount.max},`
										}



										if(!paragraph.max_discount_amount){
											paragraph.max_discount_amount = {}
										}

										if(team.data.base[paragraph.type]?.max_discount_amount.min){
											paragraph.max_discount_amount.min = `min:${team.data.base[paragraph.type]?.max_discount_amount.min},`
										}

										if(team.data.base[paragraph.type]?.max_discount_amount.max){
											paragraph.max_discount_amount.max = `max:${team.data.base[paragraph.type]?.max_discount_amount.max},`
										}



										if(!paragraph.usage_limit){
											paragraph.usage_limit = {}
										}

										if(team.data.base[paragraph.type]?.usage_limit.min){
											paragraph.usage_limit.min = `min:${team.data.base[paragraph.type]?.usage_limit.min},`
										}

										if(team.data.base[paragraph.type]?.usage_limit.max){
											paragraph.usage_limit.max = `max:${team.data.base[paragraph.type]?.usage_limit.max},`
										}



										if(!paragraph.usage_per){
											paragraph.usage_per = {}
										}

										if(team.data.base[paragraph.type]?.usage_per.min){
											paragraph.usage_per.min = `min:${team.data.base[paragraph.type]?.usage_per.min},`
										}

										if(team.data.base[paragraph.type]?.usage_per.max){
											paragraph.usage_per.max = `max:${team.data.base[paragraph.type]?.usage_per.max},`
										}



										if(!paragraph.started_at){
											paragraph.started_at = {}
										}

										if(team.data.base[paragraph.type]?.started_at.min){
											paragraph.started_at.min = `min:${team.data.base[paragraph.type]?.started_at.min},`
										}

										if(team.data.base[paragraph.type]?.started_at.max){
											paragraph.started_at.max = `max:${team.data.base[paragraph.type]?.started_at.max},`
										}



										if(!paragraph.expir){
											paragraph.expired_at = {}
										}
										if(team.data.base[paragraph.type]?.expired_at.min){
											paragraph.expired_at.min = `min:${team.data.base[paragraph.type]?.expired_at.min},`
										}

										if(team.data.base[paragraph.type]?.expired_at.max){
											paragraph.expired_at.max = `max:${team.data.base[paragraph.type]?.expired_at.max},`
										}

										paragraphs[p] = paragraph
									}
								}else{
									fallback = 'Document Not Found'

									continue
								}
							}catch(err){
								console.log('paragraphs err', err);
							}


							var contexts

							if(models['deepinfra']){
								var res = await Deepinfra(deepinfra, 'openai/gpt-oss-20b', graph2contexts(current).trim(), JSON.stringify(paragraphs))

								if(res){
									contexts = res.context
								}

								models['deepinfra'] -= 1
							}

							if(!contexts && gemini_llm_api){
								var res = await Gemini(gemini_llm_api, gemini_llm_model, graph2contexts(current).trim(), JSON.stringify(paragraphs))

								if(res){
									contexts = res.context
								}

								models[gemini_llm_api+'-'+gemini_llm_model] -= 1

							}

							if(!contexts){
								fallback = 'graph2contexts overflow'

								continue
							}


							var generation = ''

							var augmented = ''

							// 유료 회원이면 이전 컨텍스트 합쳐서 답변하기
							if(task.topK > 50){
								var { results, success, error } = await env[`${zoneRegion}_talks`].prepare(
									`SELECT * FROM talks WHERE "bcc" = "${task.bcc}" AND "created_at" < ${created_at} AND "updated_at" = ${task.updated_at} ORDER BY created_at DESC LIMIT 5`
								).all()

								if(results.length){
									for(var r = 0; r < results.length; r++){
										var retrieval = results[r]

										var { results, success, error } = await env[`${zoneRegion}_${retrieval.type}`].prepare(
											`SELECT * FROM ${retrieval.type} WHERE "ref" = "${retrieval.ref}" AND "created_at" < ${created_at} ORDER BY created_at DESC LIMIT 100`
										).all()

										var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(retrieval.data))

										var data = JSON.parse(decompressedJsonString)

										if(results.length){
											augmented += `${p}. ${data.text}\n`

											for(var r = 0; r < results.length; r++){
												var obj = safeClone(results[r])

												delete obj.from
												delete obj.to
												delete obj.cc
												delete obj.bcc
												delete obj.ref

												augmented += `${JSON.stringify(obj)}\n`
											}
										}

										statements[`${zoneRegion}_talks`].push(
											env[`${zoneRegion}_talks`].prepare(`
												UPDATE talks SET updated_at = ? WHERE id = ?
											`).bind(
												now, retrieval.id
											)
										)
									}

									if(augmented){
										augmented = `Reference Context Start\n${augmented}\nReference Context End\n`
									}
								}
							}

							if(contexts){
								if(contexts.length){
									for(var q = 0; q < contexts.length; q++){
										var context = contexts[q]

										context.id = hashId()

										if(!context.type){
											continue
										}


										var type = context.type

										if(context.type == "sales"){
											type = "sales"

											context.type = "order"

										}else if(context.type == "goods" || context.type == "order"){
											type = "sales"

										}else if(context.type == "event" || context.type == "coupon"){
											type = "event"

										}


										context.by = "created_at"

										if(context.substantial){
											context.by = context.substantial
										}


										context.sort = "DESC"

										if(context.find){
											if(find == 'light' || find == 'few' || find == 'little'){
												context.sort = "ASC"
											}
										}


										var query = {
											options:{
												topK: task.topK,
												returnValues: false, // true 이며 벡터 값 포함
												returnMetadata: true,
												filter : {
													type : context.type,
													to : team.id
												}
											}
										}

										var queryVector

										if(models['cloudflare']){
											var { data: queryVector } = await env.AI.run('@cf/baai/bge-m3', {
												text: [context.text],
											})

											models['cloudflare'] -= 1

										}

										if(!queryVector && models['deepinfra']){
											var queryVector = await Deepinfra(deepinfra, 'BAAI/bge-m3', '', context.text)

											var $VectorizeVector: VectorizeVector[] = embeddings.map((values, i) => {
												return {
													id: item.id,
													values: values,
													metadata: metadata
												}
											})

											models['deepinfra'] -= 1

										}

										if(!queryVector){
											fallback = 'overflow'

											continue
										}


										var condition = `"created_at" < ${now}`


										if(context.status){
											if(type == "sales"){
												if(context.status == "used" || context.status == "lease" || context.status == "rental" || context.status == "refurbish"){
													condition += ` AND "${context.status}" > 0 `
												}
											}else{
												condition += ` AND "status" = "${context.status}" `
											}
										}

										if(Object.keys(context.condition).length){
											for (var key in context.condition) {
												var value = context.condition[key]

												if (context.condition.hasOwnProperty(key)) {
													if(isNaN(value)){
														query.options.filter[key] = value

														if(key == "price"){
															if(value.currency){
																query.options.filter.currency = value.currency
															}
														}
													}else{
														condition += parseCondition(value, key, " AND ")
													}
												}
											}
										}


										var { matches } = await env[`${vectorRegion}-${type}`].query(queryVector[0], query.options)

										var rag = {
											search : {
												query : context.condition,
												sql : {},
												vector : {}
											}
										}

										var matches_condition = ''

										if(matches.length){
											for(var m = 0; m < matches.length; m++){
												var match = matches[m]

												if(matches_condition.length){
													matches_condition += ' OR '
												}

												matches_condition += `("id" = "${match.id}" AND "to" = "${team.id}" AND "created_at" < ${now})`
											}
										}

										var { results } = await env[`${zoneRegion}_${type}`].prepare(`SELECT * FROM ${type} WHERE ${matches_condition} LIMIT 100`).all()

										if(results.length){
											for(var r = 0; r < results.length; r++){
												var item = results[r]

												var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(item.data))

												var data = JSON.parse(decompressedJsonString)

												if(data){
													if(Object.keys(data).length){
														for (var name in data) {
															if (data.hasOwnProperty(name)) {
																var value = data[name]

																item[name] = value
															}
														}
													}
												}

												delete results[i].from
												delete results[i].to
												delete results[i].cc
												delete results[i].bcc
												delete results[i].ref
												delete results[i].data
											}

											rag.search.vector = {
												results : results
											}
											
										}

										var orderBy = ''

										if(context.sort && context.by){
											orderBy = `ORDER BY ${context.by} ${context.sort}`
										}

										var { results } = await env[`${zoneRegion}_${type}`].prepare(`SELECT * FROM ${type} WHERE ${condition} AND "to" = "${team.id}" AND "created_at" < ${now} ${orderBy} LIMIT 300`).all()

										if(results.length){
											for(var r = 0; r < results.length; r++){
												var item = results[r]

												var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(item.data))

												var data = JSON.parse(decompressedJsonString)

												if(data){
													if(Object.keys(data).length){
														for (var name in data) {
															if (data.hasOwnProperty(name)) {
																var value = data[name]

																item[name] = value
															}
														}
													}
												}

												delete results[i].from
												delete results[i].to
												delete results[i].cc
												delete results[i].bcc
												delete results[i].ref
												delete results[i].data
											}

											rag.search.sql = {
												results : results
											}
										}


										var system = 'Return the content related to the {search.text} value from the search results in a JSON structure.'

										var content = context2results(context, [...rag.search.sql.results, ...rag.search.vector.results], language)


										var text

										if(models['deepinfra']){
											text = await Deepinfra(deepinfra, 'openai/gpt-oss-20b', system, content)

											models['deepinfra'] -= 1

										}

										if(!text && gemini_llm_api){
											text = await Gemini(gemini_llm_api, gemini_llm_model, system, content, {"temperature": 1})

											models[gemini_llm_api+'-'+gemini_llm_model] -= 1

										}

										if(!text){
											fallback = 'overflow'

											continue
										}

										var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
											text : text,
											search : rag.search
										})), { to: 'arraybuffer' })

										context.data = arr.buffer

										statements[`${zoneRegion}_talks`].push(
											env[`${zoneRegion}_talks`].prepare(`
												INSERT INTO talks (
													"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "updated_at"
												) VALUES (
													?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10
												) ON CONFLICT (id) DO UPDATE SET
													"type" = EXCLUDED."type",
													"from" = EXCLUDED."from",
													"to" = EXCLUDED."to",
													"cc" = EXCLUDED."cc",
													"bcc" = EXCLUDED."bcc",
													"ref" = EXCLUDED."ref",
													"data" = EXCLUDED."data",
													"created_at" = EXCLUDED."created_at",
													"updated_at" = EXCLUDED."updated_at"
											`).bind(
												context.id,
												context.type,
												task.from,
												task.to,
												task.cc,
												task.bcc,
												task.id,
												context.data,
												now,
												now
											)
										)
									}
								}
							}	
						}
					}


					statements[region].push(
						env[region].prepare(`
							DELETE FROM tasks WHERE id = "${task.id}"
						`)
					)


					if(fallback){
						console.log('fallback',fallback);

						statements[`${zoneRegion}_talks`].push(
							env[`${zoneRegion}_talks`].prepare(`
								INSERT INTO talks (
									"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "updated_at"
								) VALUES (
									?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10
								) ON CONFLICT (id) DO UPDATE SET
									"type" = EXCLUDED."type",
									"from" = EXCLUDED."from",
									"to" = EXCLUDED."to",
									"cc" = EXCLUDED."cc",
									"bcc" = EXCLUDED."bcc",
									"ref" = EXCLUDED."ref",
									"data" = EXCLUDED."data",
									"created_at" = EXCLUDED."created_at",
									"updated_at" = EXCLUDED."updated_at"
							`).bind(
								talk.id,
								"prompt",
								task.from,
								task.to,
								task.cc,
								task.bcc,
								talk.ref,
								null,
								now,
								now
							)
						)
					}

					talk.data = null

					if(talk.text){
						var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
							text : talk.text
						})), { to: 'arraybuffer' })

						talk.data = arr.buffer
					}

					statements[`${zoneRegion}_talks`].push(
						env[`${zoneRegion}_talks`].prepare(`
							INSERT INTO talks (
								"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "updated_at"
							) VALUES (
								?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10
							) ON CONFLICT (id) DO UPDATE SET
								"type" = EXCLUDED."type",
								"from" = EXCLUDED."from",
								"to" = EXCLUDED."to",
								"cc" = EXCLUDED."cc",
								"bcc" = EXCLUDED."bcc",
								"ref" = EXCLUDED."ref",
								"data" = EXCLUDED."data",
								"created_at" = EXCLUDED."created_at",
								"updated_at" = EXCLUDED."updated_at"
						`).bind(
							talk.id,
							talk.type,
							talk.from,
							talk.to,
							talk.cc,
							talk.bcc,
							talk.ref,
							talk.data,
							now,
							now
						)
					)

					statements[`${zoneRegion}_talks`].push(
						env[`${zoneRegion}_talks`].prepare(`
							UPDATE talks SET updated_at = ? WHERE id = ?
						`).bind(
							now, task.id
						)
					)


					var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify(team.data)), { to: 'arraybuffer' })

					team.data = arr.buffer

					statements[logisRegion].push(
						env[logisRegion].prepare(`
							UPDATE users SET data = ?, updated_at = ? WHERE id = ?
						`).bind(
							team.data, now, team.id
						)
					)


					for (const region in statements) {
						if (statements.hasOwnProperty(region)) {
							var batch = statements[region]

							if(batch.length){
								console.log('region',region);
								var { results, success, error } = await env[region].batch(batch)
							}
						}
					}

					return new Response(JSON.stringify({
						models : models,
						limits : limits,
						counts : pageCount
					}), {
						headers: { "Content-Type": "application/json" },
					})
				}       
			}
		}catch(err){
			console.log('err',err);
		}

		return new Response(`I'm a teapot!`, {
			status:418,
			headers: { "Content-Type": "text/html; charset=utf-8" },
		})

	}
} satisfies ExportedHandler<Env>