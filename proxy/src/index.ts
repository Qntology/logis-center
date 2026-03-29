import { Node, parseHTML } from 'linkedom'

import { gzip, ungzip } from 'pako'

import { ethers } from 'ethers'


/*
	--- 결제 타입 ---
		$user
		$team

		+++ 결제 플로우 만들어야함

	사용자가 안사용하는 벡터 DB 자동 정리하는 기능 추가하기
*/

function crc32(s) { var polynomial = arguments.length < 2 ? 0x04C11DB7 : arguments[1], initialValue = arguments.length < 3 ? 0xFFFFFFFF : arguments[2], finalXORValue = arguments.length < 4 ? 0xFFFFFFFF : arguments[3], crc = initialValue, table = [], i, j, c; function reverse(x, n) { var b = 0; while (n) { b = b * 2 + x % 2; x /= 2; x -= x % 1; n--; } return b; } for (i = 256; i >= 0; i--) { c = reverse(i, 32); for (j = 0; j < 8; j++) { c = ((c * 2) ^ (((c >>> 31) % 2) * polynomial)) >>> 0; } table[i] = reverse(c, 32); } for (i = 0; i < s.length; i++) { c = s.charCodeAt(i); if (c > 255) { throw new RangeError(); } j = (crc % 256) ^ c; crc = ((crc / 256) ^ table[j]) >>> 0; } return (crc ^ finalXORValue) >>> 0; }

var rowsTrim = function(rows, key, value){
	if(typeof key != "undefined" && typeof value != "undefined"){
		for(var r = 0; r < rows.length; r++){
			if(rows[r]){
				if(rows[r][key]){
					if(rows[r][key] == value){
						rows[r] = undefined
					}
				}
			}
		}
	}

	return rows.filter(function( row ) {
		return row !== undefined;
	})
}

const randomKey = function(){
	var key = Math.random().toString()

	return parseInt(key.replace("0.",""))
}

function Digest(text) {
	// \p{P}: 문장 부호 및 구두점 (Punctuation)
	// \p{S}: 기호 (Symbol - 화폐, 수학 기호 등)
	// \p{Z}: 공백 및 구분자 (Separator/Whitespace)
	// u 플래그 필수
	
	const regex = /[\p{P}\p{S}\p{Z}]/gu;
	
	return hashId(text.replace(regex, "").toLowerCase());
}

function normalizeNumericHomoglyphs(str) {
	if (typeof str !== 'string') return str;

	const map = {
		// 0
		'O': '0', 'o': '0', 'Ο': '0', '○': '0', '〇': '0', '０': '0', 'Ｏ': '0',
		// 1
		'I': '1', 'l': '1', '１': '1', 'Ｉ': '1', 'ｌ': '1', 'Ι': '1', '|': '1', 'ᛁ': '1',
		// 2
		'Z': '2', 'z': '2', '２': '2', 'Ƨ': '2', 'ᒿ': '2',
		// 3
		'Ɛ': '3', 'ɜ': '3', 'З': '3', 'з': '3', '３': '3',
		// 4
		'Ꮞ': '4', '４': '4',
		// 5
		'S': '5', 's': '5', '５': '5', 'ƽ': '5',
		// 6
		'b': '6', 'Ꮾ': '6', '６': '6',
		// 7
		'T': '7', '７': '7',
		// 8
		'Β': '8', 'ß': '8', '８': '8',
		// 9
		'g': '9', '９': '9', 'ǵ': '9', 'ɡ': '9'
	};

	const chars = Object.keys(map)
		.map(s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
		.join('|');

	const regex = new RegExp(chars, 'gu');

	return str.replace(regex, ch => map[ch] || ch);
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
	const isEmpty = (value) => value === null || value === undefined || value === '' || value === 0;

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

const image2json = function(region, language, type, address){
	if(type == "tracking"){
		return `convert the shipping label image to fit the dataset JSON structure. Return only the JSON structure result, no explanation.
		#region : ${region}
		#recipient_address : ${JSON.stringify(address)}
		#tracking_number is selected from the number that matches the barcode or QR code, among others, based on the #region, excluding the format of a national telephone or mobile phone number, or an order number.
		{
			tracking_number:tracking number or 운송장 번호 or 송장 번호 or 송장번호 or 등기 번호 or 등기번호 or 运单号 or 運單號 or 伝票番号 or Número de seguimiento or Numéro de suivi or Sendungsnummer or Номер накладной or Número de rastreamento or Numero di tracciamento or رقم التتبع or Số vận đơn or Nomor resi or หมายเลขติดตามพัสดุ | string,
			recipient_match : shipping label #recipient_address match. Ruled the same despite different floor levels | boolean,
			barcodes : [barcode number | string] | array,
			text : summarize the shipping label contents in ${language}. Masking the address in the summary to District-level and up. Do not mention that information is masked or partially hidden | string,
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
					"eq":yyyy-MM-ddThh:mm:ss,"lte":yyyy-MM-ddThh:mm:ss,"gte":yyyy-MM-ddThh:mm:ss
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
					"eq":yyyy-MM-ddThh:mm:ss,"lte":yyyy-MM-ddThh:mm:ss,"gte":yyyy-MM-ddThh:mm:ss
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
					"eq":yyyy-MM-ddThh:mm:ss,"lte":yyyy-MM-ddThh:mm:ss,"gte":yyyy-MM-ddThh:mm:ss
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
					"eq":yyyy-MM-ddThh:mm:ss,"lte":yyyy-MM-ddThh:mm:ss,"gte":yyyy-MM-ddThh:mm:ss
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
	# #date : The date value is set by referencing both the natural language's implied time period and the region value against the current time (${current}); it will be marked as null if a value is absent
	# #status : 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error'
	# #substantial : 'size' or 'weight' or 'shipping_fee' or 'shipping_duration' or 'sale_price' or 'supply_price' or 'low_stock_threshold' or 'discount' or 'min_order_amount' or 'max_discount_amount' or 'usage_limit' or 'usage_per' or ''
	# #find : 'many' or 'few' or 'much' or 'little' or 'heavy' or 'light' or ''`
}


const list2json = function(language){
	return `
		type:'order' or 'goods' or 'tracking' or 'review' or 'coupon' or 'event' or '',
		item:type based item CSS1 selector excluding ads,
		more:item URL includes a manage path, an administrative or edit route Link CSS1 selector,
		node:item parent list CSS1 selector excluding ads,
		next:list next button CSS1 selector,
		text:summarize the contents of the items array in ${language},
		detail:is a detail page or a detail form | boolean,
		items: [
			if (type is 'tracking' or 'review') {
				status:'start' or 'progress' or 'stop' or 'cancel' or 'return',
				id:Refer to the ID value from the link or an attribute | string,
				title:author and content | string, 
				link:URL includes a manage path, an administrative or edit route Link | string,
				registration_date:yyyy-MM-ddThh:mm:ss | string,
			}
			if (type is 'order' or 'goods') {
				status:'show' or 'progress' or 'remove' or 'hide' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
				link:URL includes a manage path, an administrative or edit route Link | string,
				id:Refer to the ID value from the link or an attribute | string,
				title:title | string, 
				sale_price:sale price | number,
				supply_price:supply price | number,
				currency:ISO 4217 Currency Code | string,
				quantity:item stock quantity | number,
				tracking_number:Tracking Number or 운송장 번호 or 运单号 or 運單號 or 伝票番号 or Número de seguimiento or Numéro de suivi or Sendungsnummer or Номер накладной or Número de rastreamento or Numero di tracciamento or رقم التتبع or Số vận đơn or Nomor resi or หมายเลขติดตามพัสดุ | string,
				registration_date:yyyy-MM-ddThh:mm:ss | string,
			}
			if (type is 'coupon' or 'event') {
				status:'show' or 'progress' or 'hide' or 'stop' or 'cancel' or 'expire' or 'complete' or 'error',
				id:Refer to the ID value from the link or an attribute | string,
				title:type based item title, 
				started_at:yyyy-MM-ddThh:mm:ss,
				expired_at:yyyy-MM-ddThh:mm:ss,
				registration_date:yyyy-MM-ddThh:mm:ss | string,
			}
		] 
	`
}


const item2json = function(type, href){
	if(type == 'tracking'){
		return ` 
			node:${type} form container CSS1 selector,
			status:{
				value:'draft' or 'progress' or 'return' or 'complete' or 'error',
				selector:selector
			},
			id:{
				value:tracking number | string,
				selector:selector
			},
			title:{
				value:${type} goods title | string,
				selector:selector
			} 
			sender_name:{
				value:sender_name | string,
				selector:selector
			},
			sender_address:{
				value:sender_address | string,
				selector:selector
			},
			sender_phone:{
				value:sender_phone | string,
				selector:selector
			},
			recipient_name:{
				value:recipient_name | string,
				selector:selector
			},
			recipient_address:{
				value:recipient_address | string,
				selector:selector
			},
			recipient_phone:{
				value:recipient_phone | string,
				selector:selector
			},
			package_width:{
				value:Package width | number,
				selector:selector
			},
			package_height:{
				value:Package height | number,
				selector:selector
			},
			package_length:{
				value:Package length | number,
				selector:selector
			},
			package_weight:{
				value:Package weight | number,
				selector:selector
			},
			carrier:{
				value:carrier name translated into English | string,
				selector:selector
			},
			shipping_fee:{
				value:Shipping cost | number,
				selector:selector
			},
			shipping_method:{
				value:'standard' or 'express' or 'same_day' or 'pick_up' or 'freight' or 'prepaid',
				selector:selector
			},
			shipping_duration:{
				value:Estimated delivery days | number,
				selector:selector
			},
			bundle_shipping:{
				value:Allow combined shipping | string,
				selector:selector
			},
			shipping_date:{
				value:yyyy-MM-ddThh:mm:ss | string,
				selector:selector
			},
			registration_date:{
				value:yyyy-MM-ddThh:mm:ss | string,
				selector:selector
			},
		`
	}else if(type == 'goods'){
		return `
			node:${type} form container CSS1 selector,
			code:{
				value:product constant code | string,
				selector:selector
			},
			link:'${href}',
			id:{
				value:Refer to the ID value from the link or an attribute or input value | string,
				selector:selector
			},
			status:{
				value:'draft' or 'show' or 'hide' or 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
				selector:selector
			},
			payment_method:{
				value:payment method | string,
				selector:selector
			},
			bank:{
				value:bank company name or '' | string,
				selector:selector
			},
			card:{
				value:card company name or '' | string,
				selector:selector
			},
			model_name:{
				value:product Model name | string,
				selector:selector
			},
			brand_name:{
				value:product Brand name | string,
				selector:selector
			},
			condition:{
				value:['new' or 'used' or 'lease' or 'rental' or 'refurbish'],
				selector:selector
			},
			description:{
				value:product Full description (HTML allowed) | string,
				selector:selector
			},
			short_description:{
				value:product short description | string,
				selector:selector
			},
			tags:{
				value:[{ tag : product keyword or tag | string }],
				selector:selector
			},
			origin_country:{
				value:product Country of origin/manufacture | string,
				selector:selector
			},
			manufacturer:{
				value:product Manufacturer name | string,
				selector:selector
			},
			release_date:{
				value:Product release date(yyyy-MM-ddThh:mm:ss) | string,
				selector:selector
			},
			manufacture_date:{
				value:product Date(yyyy-MM-ddThh:mm:ss) of manufacture | string,
				selector:selector
			},
			expiration_date:{
				value:product Expiration or use-by date(yyyy-MM-ddThh:mm:ss) | string,
				selector:selector
			},
			gtin:{
				value:product Global Trade Item Number | string,
				selector:selector
			},
			mpn:{
				value:product Manufacturer Part Number | string,
				selector:selector
			},
			barcode:{
				value:product Barcode value | string,
				selector:selector
			},
			sale_price:{
				value:product sale price | number,
				selector:selector
			},
			supply_price:{
				value:product supply price | number,
				selector:selector
			},
			currency:{
				value:ISO 4217 Currency Code | string,
				selector:selector
			},
			compare_at_price:{
				value:product Original price for showing discounts | number,
				selector:selector
			},
			quantity:{
				value:product Inventory quantity | number,
				selector:selector
			},
			stock_keeping_unit:{
				value:Stock Keeping Unit | string,
				selector:selector
			},
			low_stock_threshold:{
				value:product Low stock alert threshold | number,
				selector:selector
			},
			unit:{
				value:product Selling unit | string,
				selector:selector
			},
			tax_included:{
				value:product Whether tax | number,
				selector:selector
			},
			tax_code:{
				value:product Tax code for region-specific rules | string,
				selector:selector
			},
			main_image_url:{
				value:Main product image URL | string,
				selector:selector
			},
			additional_image_url:{
				value:additional product image URL | string,
				selector:selector
			},
			video_url:{
				value:product Promotional video URL | string,
				selector:selector
			},
			carrier:{
				value:product carrier name translated into English | string,
				selector:selector
			},
			shipping_fee:{
				value:product Shipping cost | number,
				selector:selector
			},
			shipping_method:{
				value:'standard' or 'express' or 'same_day' or 'pick_up' or 'freight' or 'prepaid',
				selector:selector
			},
			shipping_duration:{
				value:product Estimated delivery days | number,
				selector:selector
			},
			bundle_shipping:{
				value:product Allow combined shipping | string,
				selector:selector
			},
			product_width:{
				value:Package width(cm) | number,
				selector:selector
			},
			product_height:{
				value:Package height(cm) | number,
				selector:selector
			},
			product_length:{
				value : Package length(cm) | number,
				selector:selector
				
			},
			product_weight:{
				value : Package weight(kg) | number,
				selector:selector
			},
			options:[
				{
					value:option name | string,
					selector:selector,
					inputs:[{
						value:option input value | string,
						selector:selector
					}]
				}
			],
			additional_goods:[
				{
					value:URL includes a manage path, an administrative or edit route product Link | string,
					selector:selector
				}
			],
			title:{
				value:product based title | string,
				selector:selector
			},
			registration_date:{
				value:yyyy-MM-ddThh:mm:ss | string,
				selector:selector
			},
		`
	}else if(type == 'order'){
		return `
			node:${type} form container CSS1 selector,
			link : '${href}',
			id:{
				value:Refer to the ID value from the link or an attribute or input value | string,
				selector:selector
			},
			tracking_number:{
				value:tracking number | string,
				selector:selector
			},
			status:{
				value:'draft' or 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
				selector:selector
			},
			goods:[{
				title:{
					value:goods title | string,
					selector:selector
				},
				link:{
					value:URL includes a manage path, an administrative or edit route goods Link | string,
					selector:selector
				},
				id:{
					value:Refer to the product no value from the link or an attribute or input value | string,
					selector:selector
				}
			}],
			sender_name:{
				value:sender_name | string,
				selector:selector
			},
			sender_address:{
				value:sender_address, Filter the addresses to District-level and up | string,
				selector:selector
			},
			sender_phone:{
				value:sender_phone | string,
				selector:selector
			},
			recipient_name:{
				value:recipient_name | string,
				selector:selector
			},
			recipient_address:{
				value:recipient_address, Filter the addresses to District-level and up | string,
				selector:selector
			},
			recipient_phone:{
				value:recipient_phone | string,
				selector:selector
			},
			bank:{
				value:bank company name | string,
				selector:selector
			},
			card:{
				value:card company name | string,
				selector:selector
			},
			order_date:{
				value:order date | string,
				selector:selector
			},
			payment_date:{
				value:payment date or '' | string,
				selector:selector
			},
			payment_method:{
				value:'C.O.D.' or 'CARD' or 'BANK' or '' | string,
				selector:selector
			},
			payment_origin:{
				value:Payment Gateway Service Name or '' | string,
				selector:selector
			},
			registration_date:{
				value:yyyy-MM-ddThh:mm:ss | string,
				selector:selector
			},
		`
	}else if(type == 'coupon' || type == 'event'){
		return `
			node:${type} container CSS1 selector,
			link : '${href}',
			id:{
				value:Refer to the ID value from the link or an attribute or input value | string,
				selector:selector
			},
			type:{
				value:'percentage' or 'fixed_amount' or 'free_shipping' or '',
				selector:selector
			},
			status:{
				value:'draft' or 'progress' or 'stop' or 'cancel' or 'expire' or 'complete' or 'error',
				selector:selector
			},
			title:{
				value:${type} title | string, 
				selector:selector
			},
			started_at:{
				value:yyyy-MM-ddThh:mm:ss | string,
				selector:selector
			},
			expired_at:{
				value:yyyy-MM-ddThh:mm:ss | string,
				selector:selector
			},
			code:{
				value:${type} code used at checkout | string,
				selector:selector
			},
			discount:{
				value:Discount value | number,
				selector:selector
			},
			quantity:{
				value:${type} quantity | number
				selector:selector
			},
			usage_limit:{
				value:Total usage limit for the coupon | number,
				selector:selector
			},
			usage_per:{
				value:Usage limit per customer | number,
				selector:selector
			},
			new_customer_only:{
				value:new customer only | boolean
				selector:selector
			},
			min_order_amount:{
				value:Minimum order amount required to apply coupon | number,
				selector:selector
			},
			max_discount_amount:{
				value:Maximum discount limit allowed for the coupon | number,
				selector:selector
			},
			region_restrictions:{
				value:region restrictions | boolean,
				selector:selector
			},
			registration_date:{
				value:yyyy-MM-ddThh:mm:ss | string,
				selector:selector
			},
		`
	}else if(type == 'review'){
		return `
			node:${type} container CSS1 selector,
			link : '${href}',
			id:Refer to the ID value from the link or an attribute or input value | string,,
			status:{
				value:'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' or 'error',
				selector:selector
			},
			name:{
				value:${type} name | string,
				selector:selector
			},
			title:{
				value:${type} item title | string, 
				selector:selector
			},
			completed:{
				value:order complete | boolean,
				selector:selector
			},
			registration_date:{
				value:yyyy-MM-ddThh:mm:ss | string,
				selector:selector
			},
		`
	}
}




const context2results = function(context, results, language){
	var condition = ''

	if(context.condition){
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
		results : [find the content corresponding to search.text in search.results],
		markdown : Please respond with the search.results and the context in ${language} formatted in Markdown
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
		'review' or 'coupon' or 'event'

		
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

	var gte = val(col,obj.gte)
	var lte = val(col,obj.lte)
	var eq = val(col,obj.eq)

	var column = col == 'date' ? 'created_at' : col

	if(!isNaN(gte) || !isNaN(lte) || !isNaN(eq)){
		if(obj.gte && obj.lte){
			condition += ` "${column}" >= ${val(col,obj.gte)} AND "${column}" <= ${val(col,obj.lte)}`
		}else if(obj.gte){
			condition += `"${column}" >= ${val(col,obj.gte)}`
		}else if(obj.lte){
			condition += `"${column}" <= ${val(col,obj.lte)}`
		}else if(obj.eq){
			condition += `"${column}" = ${val(col,obj.eq)}`
		}
	}else{
		condition = ''
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
			if (['script', 'style', 'link', 'noscript', 'iframe'].includes(tagName)) {
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

	commerce-apac1-logis_items
	commerce-apac1-logis-goods
	commerce-apac1-logis-order
	commerce-apac1-logis-tracking
	commerce-apac1-logis-event

	...

*/ 


const CenterRegion = "commerce_logis_center"

const LogisRegion = {
	// Western North America
	'us-w': 'commerce_logis_wnam',
	'ca-w': 'commerce_logis_wnam',

	// Eastern North America
	'us': 'commerce_logis_enam',
	'ca': 'commerce_logis_enam',
	'mx': 'commerce_logis_enam',
	'cu': 'commerce_logis_enam',
	'do': 'commerce_logis_enam',
	'pr': 'commerce_logis_enam',
	'jm': 'commerce_logis_enam',

	// Western Europe
	'gb': 'commerce_logis_weur',
	'ie': 'commerce_logis_weur',
	'fr': 'commerce_logis_weur',
	'de': 'commerce_logis_weur',
	'nl': 'commerce_logis_weur',
	'be': 'commerce_logis_weur',
	'lu': 'commerce_logis_weur',
	'ch': 'commerce_logis_weur',
	'at': 'commerce_logis_weur',
	'es': 'commerce_logis_weur',
	'pt': 'commerce_logis_weur',
	'it': 'commerce_logis_weur',
	'se': 'commerce_logis_weur',
	'no': 'commerce_logis_weur',
	'dk': 'commerce_logis_weur',
	'fi': 'commerce_logis_weur',

	// Eastern Europe
	'ru': 'commerce_logis_eeur',
	'pl': 'commerce_logis_eeur',
	'cz': 'commerce_logis_eeur',
	'hu': 'commerce_logis_eeur',
	'ro': 'commerce_logis_eeur',
	'bg': 'commerce_logis_eeur',
	'ua': 'commerce_logis_eeur',
	'gr': 'commerce_logis_eeur',
	'rs': 'commerce_logis_eeur',

	// Asia_Pacific
	'cn': 'commerce_logis_apac',
	'hk': 'commerce_logis_apac',
	'kr': 'commerce_logis_apac',
	'jp': 'commerce_logis_apac',
	'sg': 'commerce_logis_apac',
	'tw': 'commerce_logis_apac',
	'th': 'commerce_logis_apac',
	'vn': 'commerce_logis_apac',
	'my': 'commerce_logis_apac',
	'ph': 'commerce_logis_apac',
	'id': 'commerce_logis_apac',
	'in': 'commerce_logis_apac',
	'pk': 'commerce_logis_apac',
	'bd': 'commerce_logis_apac',

	// Oceania
	'au': 'commerce_logis_oc',
	'nz': 'commerce_logis_oc',
	'fj': 'commerce_logis_oc',
	'pg': 'commerce_logis_oc',

	// South America
	'br': 'commerce_logis_enam', // Brazil
	'ar': 'commerce_logis_enam', // Argentina
	'cl': 'commerce_logis_enam', // Chile
	'co': 'commerce_logis_enam', // Colombia
	'pe': 'commerce_logis_enam', // Peru

	// Africa
	'za': 'commerce_logis_weur', // South Africa
	'ng': 'commerce_logis_weur', // Nigeria
	'eg': 'commerce_logis_weur', // Egypt

	// Middle East
	'sa': 'commerce_logis_eeur', // Saudi Arabia
	'ae': 'commerce_logis_eeur', // United Arab Emirates
	'tr': 'commerce_logis_eeur', // Turkey
};



const tables = ['items', 'sales', 'event', 'talks', 'tracking']


const Related = function(type){
	var list = []

	if(type == "goods"){
		list = ['order','tracking','coupon','event']

	}else if(type == "order"){
		list = ['goods','tracking','coupon','event']

	}else if(type == "tracking"){
		list = ['goods','order','coupon','event']

	}else if(type == "coupon"){
		list = ['goods','event']

	}else if(type == "event"){
		list = ['goods','coupon']

	}else if(type == "review"){
		list = ['goods','coupon','event']

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

function operators(property, current){
	if(property == "date"){
		return {
			eq : "yyyy-MM-ddThh:mm:ss",
			lte : "yyyy-MM-ddThh:mm:ss",
			gte : "yyyy-MM-ddThh:mm:ss"
		}
	}else{
		return {
			eq : 0,
			lte : 0,
			gte : 0
		}
	}
}

const paragraph2propertys = function(task, team, paragraph, current){
	paragraph.region = task.flag // flag
	paragraph.status = '#status'
	paragraph.orderBy = '#orderBy'
	paragraph.find = '#find'
	paragraph.substantial = '#substantial'

	paragraph.condition = {}

	var type = ''

	if(paragraph.type == "tracking"){
		type = "tracking"

	}else if(paragraph.type == "sales"){
		type = "sales"

		paragraph.type = "order"

	}else if(paragraph.type == "goods" || paragraph.type == "order"){
		type = "sales"

	}else if(paragraph.type == "event" || paragraph.type == "coupon"){
		type = "event"

	}

	/*if(!type){
		// 매칭 없음
		continue
	}*/




			

	try{
		if(type){
			if(paragraph.condition.date){
				paragraph.condition.date = mergeNode(paragraph.condition.date, operators('date', current))
			}else{
				paragraph.condition.date = operators('date', current)
			}


			var propertys = ['price', 'quantity', 'width', 'height', 'length', 'weight', 'shipping_fee', 'shipping_duration', 'sale_price', 'supply_price', 'low_stock_threshold', 'discount', 'min_order_amount', 'max_discount_amount', 'usage_limit', 'usage_per', 'started_at', "expired_at"]

			for(var p = 0; p < propertys.length; p++){
				var property = propertys[p]

				if(!paragraph.condition[property]){
					paragraph.condition[property] = {}
				}

				if(team.data.base[paragraph.type][property].min){
					paragraph.condition[property].min = team.data.base[paragraph.type][property].min
				}

				if(team.data.base[paragraph.type][property].max){
					paragraph.condition[property].max = team.data.base[paragraph.type][property].max
				}

				if(Object.keys(paragraph.condition[property]).length){
					paragraph.condition[property] = mergeNode(paragraph.condition[property], operators(property, current))

					if(property == "price"){
						paragraph.condition[property].currency = '' //
					}
				}else{
					delete paragraph.condition[property]
				}
			}
		}else{
			paragraph = undefined
		}
	}catch(err){
		console.log('err',err);
	}
		

	return paragraph
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



function cleanNumber(str){
	if(str.indexOf("-") > -1){
		str = str.replace(/-/gi,"")
	}

	if(str.indexOf("_") > -1){
		str = str.replace(/_/gi,"")
	}

	if(str.indexOf(".") > -1){
		str = str.replace(/./gi,"")
	}

	if(str.indexOf(",") > -1){
		str = str.replace(/,/gi,"")
	}

	return str
}
// ================================
// Dirty JSON Parser (Recovery Focused)
// ================================

const DirtyJsonStats = {
    total: 0,
    success: 0,
    fail: 0,
};

/**
 * 정규화 핵심: 에러를 찾는 게 아니라 "JSON 규격으로 강제 개조" 합니다.
 */
function normalizeToJsonString(input) {
    if (typeof input !== "string") return input;

    let s = input.replace(/[\u00A0\u200B\u202F\uFEFF]/g, " ").trim();

    // 1. 백틱(``)이 들어온 경우를 대비해 처리 (사용자 제안 반영)
    // 백틱 안의 모든 내용을 쌍따옴표로 바꾸되, 내부의 진짜 쌍따옴표는 이스케이프 처리
    s = s.replace(/`([\s\S]*?)`/g, (_, inner) => `"${inner.replace(/"/g, '\\"')}"`);

    // 2. 키값 따옴표 보정 (key: -> "key":)
    // 따옴표가 있든 없든 일단 다 발라내서 쌍따옴표로 통일
    s = s.replace(/([{,])\s*([a-zA-Z0-9_]+)\s*:/g, '$1"$2":');

    // 3. 홑따옴표 값 보정 ('value' -> "value")
    // 값 내부에 쌍따옴표가 있으면 이스케이프 처리
    s = s.replace(/:\s*'([^']*)'/g, (_, inner) => `: "${inner.replace(/"/g, '\\"')}"`);

    // 4. CSS 셀렉터 등 "문자열 내부의 쌍따옴표" 강제 보호
    // "node": "form[name="frm"]" 같은 케이스 대응
    // 원리: ": " 로 시작해서 끝에 "나 ,가 나올 때까지 사이의 쌍따옴표를 찾음
    s = s.replace(/(":\s*")([\s\S]*?)("(?=\s*[,}\]])) /g, (match, open, body, close) => {
        // [ ] 내부의 " 를 \" 로 바꿈
        const fixedBody = body.replace(/="([^"]*)"/g, '=\\"$1\\"');
        return open + fixedBody + close;
    });

    // 5. Trailing Comma (후행 콤마) 제거
    s = s.replace(/,\s*([\]}])/g, "$1");

    // 6. 미종결 문자열 강제 닫기 (비정상 절단 대응)
    const openBraces = (s.match(/\{/g) || []).length;
    const closeBraces = (s.match(/\}/g) || []).length;
    if (openBraces > closeBraces) {
        s += "}".repeat(openBraces - closeBraces);
    }

    return s;
}

function parseDirtyJson(input) {
    DirtyJsonStats.total++;
    if (!input) return null;

    // 1차 시도: 원본 그대로 파싱
    try {
        const obj = JSON.parse(input);
        DirtyJsonStats.success++;
        return obj;
    } catch (e) {
        // 2차 시도: 강제 수리 후 파싱
        try {
            const normalized = normalizeToJsonString(input);
            const obj = JSON.parse(normalized);
            DirtyJsonStats.success++;
            return obj;
        } catch (e2) {
            // 마지막 수단: 정말 깨진 경우 null 반환
            DirtyJsonStats.fail++;
            console.warn("🔧 수리 실패:", e2.message);
            return null;
        }
    }
}


async function Deepinfra(key, model, system, user, config, inlineData){
	// DeepInfra API 호출
	var messages = []

	if(system){
		messages.push({ "role": "system", "content": system })
	}

	if(inlineData){
		messages.push({
			"role": "user", 
			"content": [
				{
					"type": "text",
					"text": system+user 
				},
				{
					"type": "image_url",   // 여기서 URL 입력
					"image_url": {
						"url": inlineData.data
					}
				}
			]
		})

		console.log('inlineData.data',inlineData.data.length);
	}else{
		if(user){
			messages.push({ "role": "user", "content": user })
		}
	}


	
		
	var body = {
		"model" : model,
		"messages": messages,
		"max_tokens": 15000,
		"temperature": config ? config.temperature : 0.95,
		"top_p": 1
	}

	// if(typeof config == "undefined"){
	// 	body["response_format"] = {
	// 		type : "json_object"
	// 	}
	// }


	var pathname = 'chat/completions'

	var isEmbedding = model.indexOf('google/embeddinggemma-300m') > -1

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
		var content = json.choices[0].message.content.trim();

		console.log('content',content);

		if(config){
			return content
		}

		try{
			var results = JSON.parse(content)

			return results
		}catch(err){

		}

		try{
			if(content.indexOf('```') > -1){
				content = content.replace(/```json/gi, "")
				content = content.replace(/```/gi, "")
				content = content.replace(/\n/gi,"")
				content = content.trim()
			}

			var results = JSON.parse(content)

			return results
		}catch(err){
			
		}

		try{
			content = content.replace(/```json/gi, "")
			content = content.replace(/```/gi, "")
			content = content.replace(/\n/gi,"")
			content = content.trim()
			content = parseDirtyJson(content)

			var results = JSON.parse(content)

			return results
		}catch(err){

		}

		return content
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
			var results = JSON.parse(content)

			return results
		}catch(err){

		}

		try{
			if(content.indexOf('```') > -1){
				content = content.replace(/```json/gi, "")
				content = content.replace(/```/gi, "")
				content = content.replace(/\n/gi,"")
				content = content.trim()
			}

			var results = JSON.parse(content)

			return results
		}catch(err){
			
		}

		try{
			content = content.replace(/```json/gi, "")
			content = content.replace(/```/gi, "")
			content = content.replace(/\n/gi,"")
			content = content.trim()
			content = parseDirtyJson(content)

			var results = JSON.parse(content)

			return results
		}catch(err){

		}
	}

	return content
}



export default {
	async fetch(
		request: Request,
		env: Env,
		ctx: ExecutionContext
	): Promise<Response> {
		// task 실행

		try{
			const buffer = await request.arrayBuffer()

			if(buffer.byteLength){
				var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(buffer))

				var json = JSON.parse(decompressedJsonString)

				var now = json.now

				var gemini_llm_api = json.gemini_llm_api

				var gemini_llm_model = json.gemini_llm_model

				var deepinfra = json.deepinfra

				var current = new Date(now).toISOString()

				var created_at = now - 10000

				var limits = json.limits

				var models = json.models

				var region = json.region

				console.log('region',region);

				// console.log('json',JSON.stringify(json));

				var fallback = ''

				var statements = {}
					statements[CenterRegion] = []
					statements[region] = []

				if(!statements[logisRegion]){
					statements[logisRegion] = []
				}

				

				try{
					var { results } = await env[region].prepare(`SELECT * FROM tasks WHERE "id" = '${json.id}' AND "ref" = '${json.ref}' AND "created_at" < ${created_at} AND "updated_at" = 0 ORDER BY created_at ASC LIMIT 1`).all()


					console.log('results.length',results.length);

					var crons = safeClone(results)

					if(crons.length){
						for(var c = 0; c < crons.length; c++){
							var cron = crons[c]

							var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(cron.data))

							var task = JSON.parse(decompressedJsonString)

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

							var vectorRegion = 'commerce-logis-'+zoneRegion

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


							if(!statements[`commerce_logis_${zoneRegion}-${tables[0]}`]){
								for(var t = 0; t < tables.length; t++){
									var table = tables[t]

									if(!statements[`commerce_logis_${zoneRegion}_${table}`]){
										statements[`commerce_logis_${zoneRegion}_${table}`] = []
									}
								}
							}

							if(!limits[task.from]){
								limits[task.from] = task.rpm
							}

							// 팀 계정으로 해야함
							var { results } = await env[logisRegion].prepare(`SELECT * FROM users WHERE "type" = 'team' AND "id" = '${task.to}' AND "created_at" < ${now} LIMIT 1`).all()

							var team = results[0]

							if(team.data){
								var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(team.data))

								team.data = JSON.parse(decompressedJsonString)
							}else{
								team.data = {}
							}

							if(limits.team){
								if(limits.team.id == team.id && isDiff(team.data, limits.team.data)){
									statements = {}

									continue
								}
							}



							console.log('team.data.base.pages',JSON.stringify(team.data.base.pages));


							var { results } = await env[logisRegion].prepare(`SELECT * FROM users WHERE "type" = 'user' AND "id" = '${team.from}' AND "created_at" < ${now} LIMIT 1`).all()

							var owner = results[0]

							var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(owner.data))

							owner.data = JSON.parse(decompressedJsonString)	
							


						


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

							console.log('task.contentType',task.contentType);

							if(task.contentType.indexOf("image/") > -1){
								var inlineData = { mimeType: task.contentType, data: task.body }

								var type = talk.type = task.type

								var address = []

								if(owner.data){
									if(owner.data.sender){
										address.push(owner.data.sender);
									}
								}

								// 추후 꼭 반영해야함 프론트 백엔드 미개발되어있음
								if(team.data){
									if(team.data.address){
										for(var a = 0; a < team.data.address.length; a++){
											var addr = team.data.address[a]

											address.push(addr)
										}
									}
								}

								var system = image2json(task.flag, language, type, address)

								var item

								if(models['deepinfra']){
									item = await Deepinfra(deepinfra, 'Qwen/Qwen3.5-0.8B', system, '', null, inlineData)

									models['deepinfra'] -= 1

								}

								if(!item && gemini_llm_api){
									item = await Gemini(gemini_llm_api, gemini_llm_model, '', system, null, inlineData)

									models[gemini_llm_api+'-'+gemini_llm_model] -= 1

								}

								if(!item){
									fallback = 'overflow'

									continue
								}

								if(!item.tracking_number){
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

								item.type = type
								

								item.no = normalizeNumericHomoglyphs(item.tracking_number)

								item.no = cleanNumber(item.no)

								item.index = crc32(hashId(item.type+team.id+item.no))

								item.id = hashId(team.id+item.index)

								item.digest = ''

								if(item.title){
									item.digest = Digest(item.title)
								}


								item.from = task.from

								item.to = task.to

								item.cc = task.cc // commerce.logis.center 잡혀져 있음 

								item.bcc = task.bcc

								item.ref = task.ref

								item.created_at = now





								var statusCode = item.status = item.status ? parseStatus(item.status) : 0


								var { results } = await env[`commerce_logis_${zoneRegion}_sales`].prepare(`SELECT * FROM sales WHERE "tracking" = ${item.index} AND "to" = '${task.to}' AND "created_at" < ${now} LIMIT 1`).all()

								var sales = results

								if(sales.length){
									var _item = safeClone(results[0])

									item.cc = task.cc = _item.cc

									item.ref = task.ref = _item.ref

									delete _item.id
									delete _item.type
									delete _item.from
									delete _item.to
									delete _item.cc
									delete _item.bcc
									delete _item.data
									delete _item.created_at
									delete _item.ref

									if(item.status){
										if(_item.status){
											delete _item.status
										}
									}

									item = mergeNode(item, _item)
								}



								if(type == "tracking"){
									var { results } = await env[`commerce_logis_${zoneRegion}_tracking`].prepare(`SELECT * FROM tracking WHERE "id" = '${item.id}' AND "to" = '${task.to}' AND "created_at" < ${now} LIMIT 1`).all()

									if(!team.data.base.pages[task.cc]){
										team.data.base.pages[task.cc] = {}
									}

									if(!team.data.base.pages[task.cc][type]){
										team.data.base.pages[task.cc][type] = {
											draft : 0,
											count : 0
										}
									}

									if(!team.data.base[type]){
										team.data.base[type] = {
											draft : 0,
											count : 0
										}
									}

									if(results.length){
										var _tracking = results[0]

										delete _tracking.ref

										if(item.status){
											if(_tracking.status){
												delete _tracking.status
											}
										}

										item = mergeNode(item, _tracking)


										if(sales.length){
											var _order = sales[0]

											item.order = _order.index

											statements[`commerce_logis_${zoneRegion}_sales`].push(
												env[`commerce_logis_${zoneRegion}_sales`].prepare(`
													UPDATE sales SET updated_at = ?, status = ?, tracking = ? WHERE id = ?
												`).bind(
													now, item.status, item.index, _order.id
												)
											)

											team.data.base.pages[task.cc].order.draft--
											team.data.base.pages[task.cc].order.count++

											team.data.base.pages[task.cc].tracking.draft--
											team.data.base.pages[task.cc].tracking.count++
										}
									}else{
										if(sales.length){
											team.data.base.pages[task.cc].tracking.draft--
											team.data.base.pages[task.cc].tracking.count++
										}else{
											team.data.base.pages[task.cc].tracking.draft++
											team.data.base[type].count++

											// items 추가 해야함
										}
									}

										

									item.ref = task.ref

									item.created_at = now

									if(sales.length){
										var _item = sales[0]

										var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(_item.data))

										var _data = JSON.parse(decompressedJsonString)

										item.order = _item.index

										statements[`commerce_logis_${zoneRegion}_items`].push(
											env[`commerce_logis_${zoneRegion}_items`].prepare(`
												INSERT INTO items (
													"id", "type", "from", "to", "cc", "bcc", "ref", "digest", "data", "created_at", "updated_at"
												) VALUES (
													?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11
												) ON CONFLICT (id) DO UPDATE SET
													"type" = EXCLUDED."type",
													"from" = EXCLUDED."from",
													"to" = EXCLUDED."to",
													"cc" = EXCLUDED."cc",
													"bcc" = EXCLUDED."bcc",
													"ref" = EXCLUDED."ref",
													"digest" = EXCLUDED."digest",
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
												item.id,
												_item.digest,
												_item.data,
												_data.time,
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


										// 추후에 반품인지 아닌지 추가해야함


										var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
											no : item.no ? item.no :"",
											type : item.type,
											text : item.text,
											link : null,
											time : now
										})), { to: 'arraybuffer' })

										item.data = arr.buffer

										var metadata = {
											id: item.id,
											no: item.no ? item.no : "",
											type: item.type,
											from: task.from,
											to: task.to,
											cc: task.cc,
											bcc: task.bcc,
											ref:pageId
										}


										var embeddings

										if(models['cloudflare']){
											var { data: embeddings } = await env.AI.run('@cf/google/embeddinggemma-300m', {
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
											var embeddings = await Deepinfra(deepinfra, 'google/embeddinggemma-300m', '', item.text)

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

										statements[`commerce_logis_${zoneRegion}_items`].push(
											env[`commerce_logis_${zoneRegion}_items`].prepare(`
												INSERT INTO items (
													"id", "type", "from", "to", "cc", "bcc", "ref", "digest", "data", "created_at", "updated_at"
												) VALUES (
													?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11
												) ON CONFLICT (id) DO UPDATE SET
													"type" = EXCLUDED."type",
													"from" = EXCLUDED."from",
													"to" = EXCLUDED."to",
													"cc" = EXCLUDED."cc",
													"bcc" = EXCLUDED."bcc",
													"digest" = EXCLUDED."digest",
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
												task.digest,
												arr.buffer,
												now,
												0
											)
										)
									}


									statements[`commerce_logis_${zoneRegion}_tracking`].push(
										env[`commerce_logis_${zoneRegion}_tracking`].prepare(`
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
											item.status,
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

								const isMore = function(_page, selectors){
									var bool = false

									var selector = ''
											
									if(_page.type == "goods"){
										selector = `${selectors.title} ${selectors.sale_price}`
									}else if(_page.type == "order"){
										selector = `${selectors.tracking_number}, ${selectors.payment_method}, ${selectors.payment_origin}, ${selectors.bank}, ${selectors.card}`
									}else if(_page.type == "tracking"){
										selector = `${selectors.title}, ${selectors.id}, ${selectors.shipping_method}`
									}else if(_page.type == "event"){
										selector = `${selectors.title}, ${selectors.started_at}, ${selectors.expired_at}`
									}else if(_page.type == "coupon"){
										selector = `${selectors.title}, ${selectors.started_at}, ${selectors.expired_at}`
									}

									if(selector){
										try{

											var { document } = parseHTML(`<html><body>${task.body}</body></html>`);

											var $target = document.querySelectorAll(selector)

											if($target.length){
												bool = true
											}

										}catch(err){
											console.log('more page err',err);
										}
									}

									return bool
								}

								var isDetail = task.detail

								console.log('isDetail000',JSON.stringify(isDetail));

								try{
									var page

									var pageType = ''

									var pageLength = 0

									console.log('task.href',task.href);

									var url = new URL(task.href)

									var pageId = hashId((task.detail ? task.cc.toUpperCase() : task.cc)+url.pathname)

									console.log('pageId',pageId);

									var { results } = await env[CenterRegion].prepare(`SELECT * FROM pages WHERE "id" = '${pageId}' AND "created_at" < ${created_at} LIMIT 1`).all()

									var pages = results

									if(pages.length){
										var _page = pages[0]

										if(_page.type){
											var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(_page.data))

											var selectors = JSON.parse(decompressedJsonString)

											isDetail = isMore(_page, selectors)

											console.log('_page isDetail',JSON.stringify(isDetail));

											if(isDetail){
												pageType = _page.type

											}else if(task.ref == pageId){
												var { results } = await env[CenterRegion].prepare(`SELECT * FROM pages WHERE "id" = '${hashId(task.cc+url.pathname.toUpperCase())}' AND "created_at" < ${created_at} LIMIT 1`).all()

												if(results.length){
													var _page = results[0]

													var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(_page.data))

													var _selectors = JSON.parse(decompressedJsonString)

													isDetail = isMore(_page, _selectors)

													if(isDetail){
														pageType = _page.type
													}

													try{
														var { document } = parseHTML(`<html><body>${task.body}</body></html>`);

														var $target = document.querySelectorAll(_page.data.node)

														if($target.length){
															task.page = _page
														}
													}catch(err){
														console.log('more page err',err);
													}
												}

											}

											try{
												var { document } = parseHTML(`<html><body>${task.body}</body></html>`);

												if(document.querySelector(selectors.node)){
													console.log('selectors.node',selectors.node);
													// task.body = document.querySelector(selectors.node).innerHTML
												}

											}catch(err){
												console.log('cache page err',err);
											}
										}
									}

									if(!isDetail){
										if(task.referrer){
											var { results } = await env[CenterRegion].prepare(`SELECT * FROM pages WHERE "id" = '${task.referrer}' AND "created_at" < ${created_at} LIMIT 1`).all()

											if(results.length){
												var _page = results[0]

												if(_page.type){
													var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(_page.data))

													var selectors = JSON.parse(decompressedJsonString)

													if(selectors.node){
														isDetail = isMore(_page, selectors)

														if(isDetail){
															pageType = _page.type
														}
													}
												}
											}
										}
									}



									var content = convertHtmlToCleanPug(task.body)

									console.log('content.length',content.length);

									console.log('isDetail1',JSON.stringify(isDetail));

									if(!isDetail){
										var system = `
											Analyze the provided Pug template and return it in the following JSON format, no explanation. 
											{language:'${language}',${list2json(language)}}
										`.trim()

										if(models['deepinfra']){
											page = await Deepinfra(deepinfra, 'openai/gpt-oss-20b', system, content)

											pageType = page.type

											if(page.items){
												pageLength = page.items.length
											}

											isDetail = page.detail

											models['deepinfra'] -= 1

										}

										if(!page && gemini_llm_api){
											page = await Gemini(gemini_llm_api, gemini_llm_model, '', `
												Analyze the provided Pug template and return it in the following JSON format, no explanation. 
												{language:'${language}',${system}}
												${content}
											`)

											pageType = page.type

											if(page.items){
												pageLength = page.items.length
											}

											isDetail = page.detail

											models[gemini_llm_api+'-'+gemini_llm_model] -= 1

										}
									}


									if((!isDetail && !pageLength && pageType) || isDetail){
										var system = `
											Analyze the provided Pug template and return it in the following JSON format, no explanation. 
											# selector : sibling value based CSS1 selector
											{language:'${language}',${item2json(pageType, task.href)}}
										`.trim()

										if(models['deepinfra']){
											page = await Deepinfra(deepinfra, 'openai/gpt-oss-20b', system, content)

											if(page.node){
												page.type = pageType

												page.link = task.link

												isDetail = true
											}else{
												page = null
											}

											models['deepinfra'] -= 1

										}

										if(!page && gemini_llm_api){
											page = await Gemini(gemini_llm_api, gemini_llm_model, system, content)

											if(page.node){
												page.type = pageType

												page.link = task.link

												isDetail = true
											}else{
												page = null
											}

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
									page.bcc = hashId(page.type+(isDetail ? task.cc.toUpperCase() : task.cc))


									if(!team.data.base.pages[task.cc]){
										team.data.base.pages[task.cc] = {}
									}

									if(page.type){
										if(!team.data.base.pages[task.cc][page.type]){
											team.data.base.pages[task.cc][page.type] = {
												draft : 0,
												count : 0
											}
										}

										if(!team.data.base[page.type]){
											team.data.base[page.type] = {
												draft : 0,
												count : 0
											}
										}
									}



									var selectors = {
										type : page.type,
										text : page.text || '',
										node : page.node || '',
										list : page.list || '',
										item : page.item || '',
										more : page.more || '',
										next : page.next || '',
										link : task.link,
										time : now,
										origin : task.origin ? task.origin : ''
									}



									var detail = {
										id : hashId(pageType+task.cc.toUpperCase()+url.pathname),
										type : pageType,
										from : task.from,
										to : task.to,
										cc : task.cc,
										bcc: hashId(page.type+task.cc.toUpperCase()),
										ref: task.ref,
										digest : '',
										data:null,
										created_at:now,
										updated_at:now
									}



									console.log('isDetail2',JSON.stringify(isDetail));

									var before

									if(isDetail){
										selectors.detail = true

										talk.ref = pageId = detail.id

										detail = null

										console.log('pageId',pageId)

										var { results } = await env[CenterRegion].prepare(`SELECT * FROM pages WHERE "id" = '${pageId}' AND "created_at" < ${created_at} LIMIT 1`).all()

										var _page

										if(results.length){
											_page = results[0]

											var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(_page.data))

											_page.data = JSON.parse(decompressedJsonString)

											before = _page.data
										}
										
										var item = safeClone(page)

										for (var p in item) {
											if (item.hasOwnProperty(p)) {
												var prop = item[p]

												if(prop){
													if(typeof prop.selector != "undefined"){
														selectors[p] = prop.selector;
														item[p] = prop.value;

													}else if(typeof prop.length == 'number'){

														for(var i = 0; i < prop.length; i++){
															var option = prop[i]

															if(typeof option == "object" && Object.keys(option).length){
																for(var n in option){
																	if (option.hasOwnProperty(n)) {
																		var opt = option[n]
																		
																		if(n == "selector"){
																			selectors[`${p}`] = option.selector

																			item[p][i] = option.value

																		}else if(typeof opt == "object" && Object.keys(opt).length){
																			for(var o in opt){
																				if (opt.hasOwnProperty(o)) {
																					var obj = opt[o]

																					if(o == "selector"){
																						selectors[`${p}_${n}`] = opt.selector

																						item[p][i][n] = opt.value
																					}
																				}
																			}
																		}
																	}
																}
															}
														}
													}
												}
											}
										}


										if(_page){
											if(!_page.data.item){
												item.ref = _page.ref
											}
										}
											

										
										page.items = [item]

									}else{
										var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
											type : page.type,
											link : task.link,
											origin : task.origin,
											detail : true,
											node : true,
											item : ''
										})), { to: 'arraybuffer' })

										detail.data = arr.buffer
									}

									console.log('pageId',pageId);

									var items = []

									if(page.items){
										if(page.items.length){
											items = page.items
										}
									}


									talk.text = page.text
									

									// page.ref = task.referrer ? task.referrer : ''
									page.ref = hashId(team.id+task.cc+page.link)

									page.data = selectors

									if(before && pages.length){
										var _page = pages[0]

										var after = selectors

										try{
											var { document } = parseHTML(`<html><body>${task.body}</body></html>`);

											if(Object.keys(before).length){
												for (var selector in before) {
													if (before.hasOwnProperty(selector)) {
														if(typeof selector == "string"){
															try{
																var $before = document.querySelectorAll(`${before[selector]}`)
																var $after = document.querySelectorAll(`${after[selector]}`)

																if($before.length && !$after.length){
																	_page.data[selector] = page.data[selector] = before[selector] + ''

																}else if(!$before.length && $after.length){
																	_page.data[selector] = page.data[selector] = after[selector] + ''

																}else if($before.length && $after.length){
																	if($before.length < $after.length){
																		_page.data[selector] = page.data[selector] = before[selector] + ''
																	}else{
																		_page.data[selector] = page.data[selector] = after[selector] + ''
																	}
																	

																}

															}catch(err){
																// console.log('selector err',err);
															}
														}
													}
												}
											}
										}catch(err){
											// console.log('_page.data err',err);
										}

										// _page.data = before

										page = mergeNode(_page, page)

										selectors = page.data

									}

									console.log('page.data',JSON.stringify(page.data));


									page.id = task.page ? task.page.id : pageId

									page.digest = ''

									console.log('page.id',page.id);

									console.log('page',JSON.stringify(page));

									var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify(page.data)), { to: 'arraybuffer' })

									page.data = arr.buffer


									if(page.type){
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
										
										statements[`commerce_logis_${zoneRegion}_items`].push(
											env[`commerce_logis_${zoneRegion}_items`].prepare(`
												INSERT INTO items (
													"id", "type", "from", "to", "cc", "bcc", "ref", "digest", "data", "created_at", "updated_at"
												) VALUES (
													?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11
												) ON CONFLICT (id) DO UPDATE SET
													"type" = EXCLUDED."type",
													"from" = EXCLUDED."from",
													"to" = EXCLUDED."to",
													"cc" = EXCLUDED."cc",
													"bcc" = EXCLUDED."bcc",
													"ref" = EXCLUDED."ref",
													"digest" = EXCLUDED."digest",
													"data" = EXCLUDED."data",
													"created_at" = EXCLUDED."created_at",
													"updated_at" = EXCLUDED."updated_at"
											`).bind(
												page.id,
												'pages',
												page.from,
												page.to,
												page.cc,
												page.bcc,
												page.ref,
												page.digest,
												page.data,
												now,
												now
											)
										)
									}

										


									talk.type = page.type

									/*
										items에서 
											시작 시간
											최저가 최대가격
											최저 수량 최대 수량

											값 가져오기
									*/ 

									console.log('진입전 items');

									if(items.length){

										console.log('items',JSON.stringify(items));
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

													var { data: queryVector } = await env.AI.run('@cf/google/embeddinggemma-300m', {
														text: [semantic],
													})

													var { matches } = await env.SEMANTIC.query(queryVector[0], query.options)

										*/

										for(var i = 0; i < items.length; i++){
											var item = items[i]


											if(!item.id){
												continue
											}

											if(!page.type){
												continue
											}

											if(isDetail){
												item.link = task.link
											}

											item.type = page.type

											item.no = cleanNumber(item.id.toString())


											item.index = crc32(hashId(item.type+team.id+item.no))

											try{
												try{
													var _url = new URL(item.link)

													item.link = (_url.pathname + _url.search).toLowerCase()
												}catch(err){
													var _url = new URL(task.origin+item.link)

													item.link = (_url.pathname + _url.search).toLowerCase()
												}


												if(isDetail){
													if(!detail.page){
														detail.page = detail.id = hashId(item.type+task.cc.toUpperCase()+_url.pathname)
													}
												}

											}catch(err){

											}


											var itemType = ""

											if(item.type == "tracking"){
												itemType = "tracking"

											}else if(item.type == "goods" || item.type == "order"){
												itemType = "sales"

											}else if(item.type == "event" || item.type == "coupon"){
												itemType = "event"

											}else{
												continue
											}

											if(item.registration_date){
												console.log('item.registration_date',item.registration_date);
												item.created_at = new Date(item.registration_date).getTime()
											}else{
												item.created_at = now
											}


											var updated_at = item.updated_at = 0

											if(isDetail){
												item.updated_at = updated_at = now
											}




											item.id = hashId(team.id+item.index)

											item.flag = task.flag
											
											item.from = task.from
											item.to = task.to
											item.cc = task.cc

											item.bcc = hashId(item.type+(isDetail ? task.cc.toUpperCase() : task.cc))

											item.ref = hashId(team.id+task.cc+item.link)

											item.data = {
												id : item.id,
												no : item.no ? item.no : "",
												title : item.title ? item.title : '',
												link : item.link,
												time : now,
												origin : task.origin ? task.origin : ''
											}

											item.digest = ''

											if(item.title){
												item.digest = Digest(item.title)
											}






											var { results } = await env[`commerce_logis_${zoneRegion}_${itemType}`].prepare(`SELECT * FROM ${itemType} WHERE "id" = '${item.id}' AND "index" = ${item.index} AND "to" = '${task.to}' AND "cc" = '${task.cc}' AND "created_at" < ${now} LIMIT 1`).all()

											if(results.length == 0){
												var { results } = await env[`commerce_logis_${zoneRegion}_${itemType}`].prepare(`SELECT * FROM ${itemType} WHERE "to" = '${task.to}' AND "cc" = '${task.cc}' AND "ref" = '${item.ref}' AND "created_at" < ${now} LIMIT 1`).all()

											}

											console.log('add results.length',item.type,results.length);

											console.log('item.updated_at',item.updated_at ? 'true' : 'false');

											console.log('team.data.base.pages[task.cc][item.type]',JSON.stringify(team.data.base.pages[task.cc]));


											if(results.length){
												var _item = results[0]

												try{
													var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(_item.data))

													_item.data = JSON.parse(decompressedJsonString)

													item.no = item.data.no = _item.data.no

													item.index = crc32(hashId(item.type+team.id+item.no))

													item.id = hashId(team.id+item.index)

													if(item.data.item && !_item.data.item){
														item.data.item = _item.data.item
														item.data.node = _item.data.node
														
													}

													if(_item.updated_at){
														item.bcc = hashId(item.type+task.cc.toUpperCase())
														item.updated_at = updated_at = now
													}
												}catch(err){
													console.log('ungzip err',err, typeof _item.data);
													delete _item.data
												}

												if(item.status){
													if(_item.status){
														delete _item.status
													}
												}


												item = mergeNode(_item, item)

												var { results } = await env[`commerce_logis_${zoneRegion}_items`].prepare(`SELECT * FROM items WHERE "id" = '${_item.id}' AND "to" = '${task.to}' AND "cc" = '${task.cc}' AND "created_at" < ${now} LIMIT 1`).all()

												if(results.length){
													var _item = results[0]

													console.log('_item.updated_at',_item.updated_at);
													console.log('item.updated_at',updated_at);

													if(!_item.updated_at){
														if(updated_at){
															team.data.base.pages[task.cc][item.type].draft--
															team.data.base.pages[task.cc][item.type].count++
														}
													}
												}

											}else{
												if(updated_at){
													team.data.base.pages[task.cc][item.type].count++
												}else{
													team.data.base.pages[task.cc][item.type].draft++
												}

												team.data.base[item.type].count++
											}

											


											var goods = item.goods ? safeClone(item.goods) : []

											delete item.goods

											try{
												if(typeof goods.length != "undefined"){
													goods.unshift({})
												}else{
													goods = []
												}
											}catch(err){
												console.log('goods err',err);
											}


											item.currency = item.currency ? item.currency.toUpperCase() : ""

											item.quantity = item.quantity ? parseInt(item.quantity) : 0

											

											item.semantic = item.title

											item.started_at = item.manufacture_date ? item.manufacture_date : 0
											
											item.expired_at = item.expiration_date ? item.expiration_date : 0

											var statusCode = item.status = item.status ? parseStatus(item.status) : 0

											


											if(item.tracking_number){
												var tracking_number = normalizeNumericHomoglyphs(item.tracking_number)
													tracking_number = cleanNumber(tracking_number)

												item.tracking = crc32(hashId('tracking'+team.id+tracking_number))
											}

												

											try{
												console.log('item.type',item.type);

												if(item.type == "order" && goods){
													/*
														상세와 리스트 차이가 분명히 있음

														주문 
															리스트에서는 송장번호가 없음
															상세페이지에서는 송장번호가 있음
													*/


													if(goods.length && item.tracking_number){
														for(var g = 0; g < goods.length; g++){

															var good = safeClone(goods[g])

															var tracking = safeClone(item)

															tracking.type = "tracking"

															tracking.no = tracking_number

															tracking.index = item.tracking

															if(good.id){
																var no = cleanNumber(good.id.toString())

																good.no = no

																good.index = crc32(hashId('goods'+team.id+good.no))

																var { results } = await env[`commerce_logis_${zoneRegion}_sales`].prepare(`SELECT * FROM sales WHERE "type" = 'goods' AND "to" = '${task.to}' AND "cc" = '${task.cc}' AND "created_at" < ${now} LIMIT 1`).all()

																if(results.length){
																	tracking.event = results[0].event
																}

																tracking.goods = good.index

																tracking.id = hashId(team.id+tracking.no+good.no)

															}else{
																tracking.id = hashId(team.id+tracking.no)

															}

															tracking.digest = item.digest


															tracking.status = item.status

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
																time : now,
																origin : task.origin ? task.origin : ""
															}

															if(!team.data.base.pages[task.cc]){
																team.data.base.pages[task.cc] = {}
															}

															if(!team.data.base.pages[task.cc][tracking.type]){
																team.data.base.pages[task.cc][tracking.type] = {
																	draft : 0,
																	count : 0
																}
															}



															var { results } = await env[`commerce_logis_${zoneRegion}_tracking`].prepare(`SELECT * FROM tracking WHERE "index" = ${tracking.index} AND "to" = '${task.to}' AND "created_at" < ${now} LIMIT 1`).all()

															if(results.length){
																var _tracking = safeClone(results[0])

																var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(_tracking.data))

																_tracking.data = JSON.parse(decompressedJsonString)

																delete _tracking.ref

																tracking = mergeNode(_tracking, tracking)
															}else{
																// 처음 저장할때 자연어 LLM으로 전처리해서 벡터 저장해야함
																var content = JSON.stringify({
																	size : tracking.size ? tracking.size : "",
																	currency : tracking.currency ? tracking.currency : "",
																	carrier : tracking.carrier ? tracking.carrier : "",
																	shipping_fee : tracking.shipping_fee ? true : false,
																	shipping_method : tracking.shipping_method ? tracking.shipping_method : "",
																	fulfillment_service : tracking.fulfillment_service ? tracking.fulfillment_service : "",
																	stock_keeping_unit : tracking.stock_keeping_unit ? tracking.stock_keeping_unit : "",
																	bundle_shipping : tracking.bundle_shipping ? true : false,
																	used : tracking.used ? true : false,
																	lease : tracking.lease ? true : false,
																	rental : tracking.rental ? true : false,
																	refurbish : tracking.refurbish ? true : false,
																	tax_included : tracking.tax_included ? true : false,
																	sender_address : tracking.sender_address,
																	recipient_address : tracking.recipient_address,
																	payment_method : tracking.payment_method, 
																	payment_origin : tracking.payment_origin 
																})

																var semantic


																var system = semantic_prompt_system(language).trim()

																if(models['deepinfra']){
																	semantic = await Deepinfra(deepinfra, 'openai/gpt-oss-20b', system, content, {"temperature": 1})

																	models['deepinfra'] -= 1

																}

																if(!semantic && gemini_llm_api){
																	semantic = await Gemini(gemini_llm_api, gemini_llm_model, system, content, {"temperature": 1})

																	models[gemini_llm_api+'-'+gemini_llm_model] -= 1

																}

																if(!semantic){
																	fallback = 'semantic overflow'

																	continue
																}

																var metadata = {
																	type: item.type,
																	from: item.from,
																	to: item.to,
																	cc: item.cc,
																	bcc: item.bcc,
																	ref:pageId
																}

																var embeddings

																if(models['cloudflare']){
																	var { data: embeddings } = await env.AI.run('@cf/google/embeddinggemma-300m', {
																		text: [semantic]
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
																	var embeddings = await Deepinfra(deepinfra, 'google/embeddinggemma-300m', '', semantic.tirm())

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
																	fallback = 'embeddings overflow'

																	continue
																}


																team.data.base.pages[task.cc][tracking.type].draft++
																// team.data.base.pages[task.cc][tracking.type].count--
																team.data.base[item.type].count++

																await env[`${vectorRegion}-${itemType}`].upsert($VectorizeVector)
															}

															var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify(tracking.data)), { to: 'arraybuffer' })

															console.log('tracking',JSON.stringify(tracking))

															statements[`commerce_logis_${zoneRegion}_tracking`].push(
																env[`commerce_logis_${zoneRegion}_tracking`].prepare(`
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
																	tracking.status,
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


															statements[`commerce_logis_${zoneRegion}_items`].push(
																env[`commerce_logis_${zoneRegion}_items`].prepare(`
																	INSERT INTO items (
																		"id", "type", "from", "to", "cc", "bcc", "ref", "digest", "data", "created_at", "updated_at"
																	) VALUES (
																		?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11
																	) ON CONFLICT (id) DO UPDATE SET
																		"type" = EXCLUDED."type",
																		"from" = EXCLUDED."from",
																		"to" = EXCLUDED."to",
																		"cc" = EXCLUDED."cc",
																		"bcc" = EXCLUDED."bcc",
																		"ref" = EXCLUDED."ref",
																		"digest" = EXCLUDED."digest",
																		"data" = EXCLUDED."data",
																		"created_at" = EXCLUDED."created_at",
																		"updated_at" = EXCLUDED."updated_at"
																`).bind(
																	tracking.id,
																	tracking.type,
																	tracking.from,
																	tracking.to,
																	tracking.cc,
																	tracking.bcc,
																	tracking.ref,
																	tracking.digest,
																	arr.buffer,
																	now,
																	updated_at
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
													item.lease = 2
												}

												if(item.condition.indexOf('rental') > -1){
													item.rental = 3
												}

												if(item.condition.indexOf('refurbish') > -1){
													item.refurbish = 4
												}
											}


											var { results } = await env[`commerce_logis_${zoneRegion}_${itemType}`].prepare(`SELECT * FROM ${itemType} WHERE "id" = '${item.id}' AND "to" = '${task.to}' AND "created_at" < ${now} LIMIT 1`).all()

											if(results.length){
												var _item = results[0]

												delete _item.ref

												if(item.status){
													if(_item.status){
														delete _item.status
													}
												}

												item = mergeNode(_item, item)


											}else{
												var { results } = await env[`commerce_logis_${zoneRegion}_${itemType}`].prepare(`SELECT * FROM ${itemType} WHERE "index" = ${item.index} AND "to" = '${task.to}' AND "cc" = '${task.cc}' AND "created_at" < ${now} LIMIT 1`).all()

												if(results.length){
													var _item = results[0]

													delete _item.ref

													if(item.status){
														if(_item.status){
															delete _item.status
														}
													}

													item = mergeNode(_item, item)
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
											
											if(related.length){
												for(var r = 0; r < related.length; r++){
													var { query, merge } = Relay(related[r], item)

													// console.log('query, merge',query, merge);

													// flow ${type}에 ${column} foreign 값이 없으면 업데이트 해야함

													if(!query || !merge){
														continue
													}

													try{
														if(query.length){
															var table = query[0].table
															var type = query[0].type
															var column = query[0].column
															var column_value = query[0].value
															var status = query[0].status

															// if(typeof status != "undefined"){
															// 	var { results } = await env[`commerce_logis_${zoneRegion}_${table}`].prepare(
															// 		`SELECT * FROM ${table} WHERE "type" = "${type}" AND "${column}" = ? AND "to" = ? AND "cc" = ? AND "status" < ? AND "created_at" < ? ORDER BY created_at DESC LIMIT 1`
															// 	).bind(
															// 		column_value, team.id, item.cc, status, now
															// 	).all()
															// }else{
															// 	var { results } = await env[`commerce_logis_${zoneRegion}_${table}`].prepare(
															// 		`SELECT * FROM ${table} WHERE "type" = "${type}" AND "${column}" = ? AND "to" = ? AND "cc" = ? AND "created_at" < ? ORDER BY created_at DESC LIMIT 1`
															// 	).bind(
															// 		column_value, team.id, item.cc, now
															// 	).all()
															// }

															var { results } = await env[`commerce_logis_${zoneRegion}_${table}`].prepare(
																`SELECT * FROM ${table} WHERE "type" = ? AND "${column}" = ? AND "to" = ? AND "cc" = ? AND "created_at" < ? ORDER BY created_at DESC LIMIT 1`
															).bind(
																type, column_value, team.id, item.cc, now
															).all()


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
																			merge = relate.merge.update
																		}

																		if(relate.merge.upsert){
																			merge = relate.merge.upsert
																		}

																		if(!merge){
																			continue
																		}

																		var foreign = merge.foreign

																		var node = {}

																		var from = row.type == merge.from ? item : row

																		var to = row.type == merge.to ? item : row

																		if(row.type == merge.to){
																			to.updated_at = updated_at
																		}
																		
																		if(merge.includes){
																			if(merge.includes.length){
																				for(var v = 0; v < merge.includes.length; v++){
																					var include = merge.includes[v]

																					node[include] = from[include]
																				}
																			}
																		}

																		


																		try{
																			var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(from.data))

																			var data = JSON.parse(decompressedJsonString)

																		}catch(err){
																			console.log('relate err',err);
																		}
																		

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

																		if(to.id == item.id){
																			item = edge
																		}


																		

																		
																		if(relate.type == merge.from){
																			// import

																			if(from.type == "goods" && to.type == "order"){
																				var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
																					id : edge.id,
																					no : edge.no ? edge.no : "",
																					title : edge.title,
																					link : edge.link,
																					time : now,
																					origin : data.origin ? data.origin : "",
																					data : data
																				})), { to: 'arraybuffer' })

																				// edge.vectorize = true

																				// edge.data = arr.buffer

																				var metadata = {
																					title : data.title ? data.title : "",
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

																				var system = semantic_prompt_system(language).trim()

																				if(models['deepinfra']){
																					semantic = await Deepinfra(deepinfra, 'openai/gpt-oss-20b', system, content, {"temperature": 1})

																					models['deepinfra'] -= 1

																				}

																				if(!semantic && gemini_llm_api){
																					semantic = await Gemini(gemini_llm_api, gemini_llm_model, system, content, {"temperature": 1})

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
																				metadata.bcc = edge.bcc
																				metadata.ref = edge.ref

																				var embeddings

																				var $VectorizeVector

																				if(models['cloudflare']){
																					var { data: embeddings } = await env.AI.run('@cf/google/embeddinggemma-300m', {
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
																					var embeddings = await Deepinfra(deepinfra, 'google/embeddinggemma-300m', '', semantic.tirm())

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


																		var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(edge.data))

																		edge.data = JSON.parse(decompressedJsonString)


																		edge.updated_at = edge.data.item ? 0 : now



																		if(edgeType == "sales"){
																			statements[`commerce_logis_${zoneRegion}_sales`].push(
																				env[`commerce_logis_${zoneRegion}_sales`].prepare(`
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
																			statements[`commerce_logis_${zoneRegion}_tracking`].push(
																				env[`commerce_logis_${zoneRegion}_tracking`].prepare(`
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
																			statements[`commerce_logis_${zoneRegion}_event`].push(
																				env[`commerce_logis_${zoneRegion}_event`].prepare(`
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



																		var { results } = await env[`commerce_logis_${zoneRegion}_items`].prepare(`SELECT * FROM items WHERE "id" = '${edge.id}' AND "created_at" < ${now} AND "updated_at" = 0 LIMIT 1`).all()

																		if(results.length){
																			var _item = results[0]

																			if(!_item.updated_at){
																				if(edge.updated_at){
																					// before ${type}에 ${column} index 값이 없으면 업데이트 해야함
																					team.data.base.pages[edge.cc][edge.type].draft--
																					team.data.base.pages[edge.cc][edge.type].count++

																					statements[`commerce_logis_${zoneRegion}_items`].push(
																						env[`commerce_logis_${zoneRegion}_items`].prepare(`
																							UPDATE items SET updated_at = ? WHERE id = ?
																						`).bind(
																							now, row.id
																						)
																					)
																				}
																			}
																		}else{
																			team.data.base.pages[edge.cc][edge.type].draft++
																			team.data.base[edge.type].count++
																		}


																			

																		// for end
																	}

																	// if end
																}else{

																	/*
																		items로 하지 말고
																			type 테이블에서 index로 있는지 여부 찾기
																	*/
																	// var draftId = hashId(item.id)


																	var { results } = await env[`commerce_logis_${zoneRegion}_${edgeType}`].prepare(`SELECT * FROM ${edgeType} WHERE "index" = '${edge.index}' AND "type" = '${edge.type}' AND "created_at" < ${now} LIMIT 1`).all()

																	if(results.length){
																		var _item = results[0]

																		if(!_item.updated_at){
																			if(updated_at){
																				team.data.base.pages[edge.cc][edge.type].draft--
																				team.data.base.pages[edge.cc][edge.type].count++
																				team.data.base[edge.type].count++

																				statements[`commerce_logis_${zoneRegion}_items`].push(
																					env[`commerce_logis_${zoneRegion}_items`].prepare(`
																						UPDATE items SET updated_at = ? WHERE id = ?
																					`).bind(
																						now, item.id
																					)
																				)
																			}
																		}
																	}else{
																		team.data.base.pages[edge.cc][edge.type].draft++
																		team.data.base[edge.type].count++

																		var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
																			id : edge.id,
																			no : item.no ? item.no : "",
																			title : item.title,
																			link : item.link,
																			time : now,
																			data : relate
																		})), { to: 'arraybuffer' })

																		statements[`commerce_logis_${zoneRegion}_items`].push(
																			env[`commerce_logis_${zoneRegion}_items`].prepare(`
																				INSERT INTO items (
																					"id", "type", "from", "to", "cc", "bcc", "ref", "digest", "data", "created_at", "updated_at"
																				) VALUES (
																					?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11
																				) ON CONFLICT (id) DO UPDATE SET
																					"type" = EXCLUDED."type",
																					"from" = EXCLUDED."from",
																					"to" = EXCLUDED."to",
																					"cc" = EXCLUDED."cc",
																					"bcc" = EXCLUDED."bcc",
																					"ref" = EXCLUDED."ref",
																					"digest" = EXCLUDED."digest",
																					"data" = EXCLUDED."data",
																					"created_at" = EXCLUDED."created_at",
																					"updated_at" = EXCLUDED."updated_at"
																			`).bind(
																				edge.id,
																				type,
																				item.from,
																				item.to,
																				item.cc,
																				item.bcc,
																				'',
																				item.digest,
																				arr.buffer,
																				now,
																				0
																			)
																		)
																	}





																}
															}
														}
													}

													// for end
												}

												// if end
											}

											console.log('typeof item.vectorize',typeof item.vectorize);

											if(item.semantic && !item.vectorize){
												var metadata = {
													type: item.type,
													from: item.from,
													to: item.to,
													cc: item.cc,
													bcc: item.bcc,
													ref:pageId
												}

												var embeddings

												if(models['cloudflare']){
													var { data: embeddings } = await env.AI.run('@cf/google/embeddinggemma-300m', {
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
													var embeddings = await Deepinfra(deepinfra, 'google/embeddinggemma-300m', '', item.semantic.tirm())

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

											item.data.text = item.semantic
											item.data.ref = page.ref

											console.log('item',JSON.stringify(item))

											var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify(item.data)), { to: 'arraybuffer' })

											item.data = arr.buffer

											statements[`commerce_logis_${zoneRegion}_items`].push(
												env[`commerce_logis_${zoneRegion}_items`].prepare(`
													INSERT INTO items (
														"id", "type", "from", "to", "cc", "bcc", "ref", "digest", "data", "created_at", "updated_at"
													) VALUES (
														?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11
													) ON CONFLICT (id) DO UPDATE SET
														"type" = EXCLUDED."type",
														"from" = EXCLUDED."from",
														"to" = EXCLUDED."to",
														"cc" = EXCLUDED."cc",
														"bcc" = EXCLUDED."bcc",
														"ref" = EXCLUDED."ref",
														"digest" = EXCLUDED."digest",
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
													item.digest,
													item.data,
													now,
													updated_at
												)
											)
												
											if(itemType == "sales"){
												statements[`commerce_logis_${zoneRegion}_sales`].push(
													env[`commerce_logis_${zoneRegion}_sales`].prepare(`
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
														item.status,
														parseFloat(item.width ? item.width : 0),
														parseFloat(item.height ? item.height : 0),
														parseFloat(item.length ? item.length : 0),
														parseFloat(item.weight ? item.weight : 0),
														item.size ? item.size : "",
														item.currency ? item.currency : "",
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
												statements[`commerce_logis_${zoneRegion}_tracking`].push(
													env[`commerce_logis_${zoneRegion}_tracking`].prepare(`
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
														item.status,
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
												statements[`commerce_logis_${zoneRegion}_event`].push(
													env[`commerce_logis_${zoneRegion}_event`].prepare(`
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
														item.status,
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

									if(detail && page.type){
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
												detail.id,
												detail.type,
												detail.from,
												detail.to,
												detail.cc,
												detail.bcc,
												detail.ref,
												detail.data,
												now,
												now
											)
										)

										statements[`commerce_logis_${zoneRegion}_items`].push(
											env[`commerce_logis_${zoneRegion}_items`].prepare(`
												INSERT INTO items (
													"id", "type", "from", "to", "cc", "bcc", "ref", "digest", "data", "created_at", "updated_at"
												) VALUES (
													?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11
												) ON CONFLICT (id) DO UPDATE SET
													"type" = EXCLUDED."type",
													"from" = EXCLUDED."from",
													"to" = EXCLUDED."to",
													"cc" = EXCLUDED."cc",
													"bcc" = EXCLUDED."bcc",
													"ref" = EXCLUDED."ref",
													"digest" = EXCLUDED."digest",
													"data" = EXCLUDED."data",
													"created_at" = EXCLUDED."created_at",
													"updated_at" = EXCLUDED."updated_at"
											`).bind(
												detail.id,
												'pages',
												detail.from,
												detail.to,
												detail.cc,
												detail.bcc,
												detail.ref,
												detail.digest,
												detail.data,
												now,
												now
											)
										)
									}

									task.title = page.text // Analyze the provided Pug template and return it in the following JSON format

									task.semantic = page.text

									if(items.length){
										var type = talk.type = page.type

										if(page.type == "sales"){
											type = "sales"

										}else if(page.type == "goods" || page.type == "order"){
											type = "sales"

										}else if(page.type == "event" || page.type == "coupon"){
											type = "event"

										}


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
										time : now,
										origin : task.origin ? task.origin : ''
									})), { to: 'arraybuffer' })

									statements[`commerce_logis_${zoneRegion}_items`].push(
										env[`commerce_logis_${zoneRegion}_items`].prepare(`
											INSERT INTO items (
												"id", "type", "from", "to", "cc", "bcc", "ref", "digest", "data", "created_at", "updated_at"
											) VALUES (
												?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11
											) ON CONFLICT (id) DO UPDATE SET
												"type" = EXCLUDED."type",
												"from" = EXCLUDED."from",
												"to" = EXCLUDED."to",
												"cc" = EXCLUDED."cc",
												"bcc" = EXCLUDED."bcc",
												"ref" = EXCLUDED."ref",
												"digest" = EXCLUDED."digest",
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
											task.digest,
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

								// team.data.base['prompt'].count++

								var paragraphs

								var system = para2graph(language)

								if(models['deepinfra']){
									var res = await Deepinfra(deepinfra, 'openai/gpt-oss-20b', system, task.body)

									if(res){
										paragraphs = res.context
									}

									models['deepinfra'] -= 1

								}

								if(!paragraphs && gemini_llm_api){
									var res = await Gemini(gemini_llm_api, gemini_llm_model, system, task.body)

									if(res){
										paragraphs = res.context
									}

									models[gemini_llm_api+'-'+gemini_llm_model] -= 1

								}

								if(!paragraphs){
									fallback = 'para2graph overflow'

									continue
								}

								console.log('before paragraphs',JSON.stringify(paragraphs))


								try{
									if(paragraphs.length){
										for(var p = 0; p < paragraphs.length; p++){
											var paragraph = paragraphs[p]

											paragraphs[p] = paragraph2propertys(task, team, paragraph, current)
										}
									}else{
										fallback = 'paragraph2propertys'
										// 올바르지 않은 대화요청입니다 리턴해야함

										continue
									}
								}catch(err){
									console.log('paragraphs err', err);
								}


								if(paragraphs.length){
									paragraphs = rowsTrim(paragraphs)
								}

								console.log('after paragraphs',JSON.stringify(paragraphs))

								if(paragraphs.length == 0){
									continue
								}


								var contexts

								var system = graph2contexts(current)

								if(models['deepinfra']){
									var results = await Deepinfra(deepinfra, 'openai/gpt-oss-20b', system, `{contexts : ${JSON.stringify(paragraphs)}}`)

									console.log('contexts res',JSON.stringify(results))

									if(results){
										if(results.contexts){
											contexts = results.contexts
										}else if(Object.keys(results).length){
											if(!results.text){
												results.text = task.body
											}

											if(!results.type && paragraphs.length == 1){
												results.type = paragraphs[0].type
											}


											contexts = [results]
										}
									}

									models['deepinfra'] -= 1
								}

								// if(!contexts && gemini_llm_api){
								// 	var res = await Gemini(gemini_llm_api, gemini_llm_model, system, `{contexts : ${JSON.stringify(paragraphs)}}`)

								// 	if(res){
								// 		if(res.contexts){
								// 			contexts = res.contexts
								// 		}
								// 	}

								// 	models[gemini_llm_api+'-'+gemini_llm_model] -= 1

								// }


								if(!contexts){
									fallback = 'graph2contexts overflow'

									continue
								}


								console.log('contexts',JSON.stringify(contexts))




								var augmented = ''

								// // 유료 회원이면 이전 컨텍스트 합쳐서 답변하기
								// if(task.topK > 50){
								// 	var { results, success, error } = await env[`commerce_logis_${zoneRegion}_talks`].prepare(
								// 		`SELECT * FROM talks WHERE "bcc" = '${task.bcc}' AND "created_at" < ${created_at} AND "updated_at" = ${task.updated_at} ORDER BY created_at DESC LIMIT 5`
								// 	).all()

								// 	if(results.length){
								// 		for(var r = 0; r < results.length; r++){
								// 			var retrieval = results[r]

								// 			var { results, success, error } = await env[`commerce_logis_${zoneRegion}_${retrieval.type}`].prepare(
								// 				`SELECT * FROM ${retrieval.type} WHERE "ref" = '${retrieval.ref}' AND "created_at" < ${created_at} ORDER BY created_at DESC LIMIT 100`
								// 			).all()

								// 			var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(retrieval.data))

								// 			var data = JSON.parse(decompressedJsonString)

								// 			if(results.length){
								// 				augmented += `${p}. ${data.text}\n`

								// 				for(var b = 0; b < results.length; b++){
								// 					var obj = safeClone(results[b])

								// 					delete obj.from
								// 					delete obj.to
								// 					delete obj.cc
								// 					delete obj.bcc
								// 					delete obj.ref

								// 					augmented += `${JSON.stringify(obj)}\n`
								// 				}
								// 			}

								// 			statements[`commerce_logis_${zoneRegion}_talks`].push(
								// 				env[`commerce_logis_${zoneRegion}_talks`].prepare(`
								// 					UPDATE talks SET updated_at = ? WHERE id = ?
								// 				`).bind(
								// 					now, retrieval.id
								// 				)
								// 			)
								// 		}

								// 		if(augmented){
								// 			augmented = `Reference Context Start\n${augmented}\nReference Context End\n`
								// 		}
								// 	}
								// }

								if(contexts){
									if(contexts.length){
										for(var q = 0; q < contexts.length; q++){
											var context = contexts[q]

											context.id = hashId()

											if(!context.type){
												continue
											}

											if(context.date == '#date'){
												context.date = ''
											}

											if(context.status == '#status'){
												context.status = ''
											}

											if(context.substantial == '#substantial'){
												context.substantial = ''
											}

											if(context.find == '#find'){
												context.find = ''
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
												if(context.find == 'light' || context.find == 'few' || context.find == 'little'){
													context.sort = "ASC"
												}
											}


											var query = {
												options:{
													topK: task.topK,
													returnValues: true, // true 이며 벡터 값 포함
													returnMetadata: 'all',
													filter : {
														type : context.type,
														to : team.id
													}
												}
											}

											var queryVector

											console.log('context.text',context.text);

											if(models['cloudflare']){
												var embeddings = await env.AI.run('@cf/google/embeddinggemma-300m', {
													text: [context.text],
												})

												queryVector = embeddings.data[0];

												models['cloudflare'] -= 1

											}


											if(!queryVector && models['deepinfra']){
												var embeddings = await Deepinfra(deepinfra, 'google/embeddinggemma-300m', '', context.text)

												queryVector = embeddings

												models['deepinfra'] -= 1

											}

											if(!queryVector){
												fallback = 'overflow'

												continue
											}




											var condition = ` "type" = '${context.type}' `

											if(context.status){
												if(type == "sales"){
													if(context.status == "used" || context.status == "lease" || context.status == "rental" || context.status == "refurbish"){
														condition += ` AND "${context.status}" > 0 `
													}
												}else{
													var status = parseStatus(context.status)

													if(status){
														condition += ` AND "status" = ${status} `
													}
												}
											}

											var temp = {}

											/*
												{
													"date":{"gte":"2025-06-01T00:00:00","lte":"2025-08-31T23:59:59"},
													"quantity":{"max":1},
													"sale_price":{"max":615600}
												}
											*/										

											if(Object.keys(context.condition).length){
												for (var key1 in context.condition) {
													var obj = context.condition[key1]

													if (context.condition.hasOwnProperty(key1)) {
														for (var key2 in obj) {
															var value = obj[key2]

															if(value){
																if(key2 == "min" || key2 == "max"){

																}else{
																	if(!temp[key1]){
																		temp[key1] = true

																		if(isNaN(value)){
																			if(key1 != "date"){
																				query.options.filter[key1] = value
																			}

																			condition += parseCondition(obj, key1, " AND ")

																			if(key1 == "price"){
																				if(context.condition.currency){
																					query.options.filter.currency = context.condition.currency
																				}
																			}

																		}else{
																			condition += parseCondition(value, key1, " AND ")
																		}
																	}
																}
															}
																
														}
													}
												}
											}

											console.log('before condition',condition);

											// if(condition){
											// 	condition = condition.replace(' AND ', '')
											// 	condition = condition.trim()
											// }

											console.log('context.condition',JSON.stringify(context.condition));

											console.log('query.options',JSON.stringify(query.options))
											console.log(`vectorRegion type ${vectorRegion}-${type}`);

											var { matches } = await env[`${vectorRegion}-${type}`].query(queryVector, query.options)

											var rag = {
												search : {
													query : context.condition,
													sql : {
														results : []
													},
													vector : {
														results : []
													}
												}
											}

											rag.search.sql

											console.log('matches.length',matches.length.toString());

											var matches_condition = ''

											if(matches.length){
												for(var m = 0; m < matches.length; m++){
													var match = matches[m]

													if(matches_condition.length){
														matches_condition += ' OR '
													}

													matches_condition += `("id" = '${match.id}' AND "to" = '${team.id}' AND "created_at" < ${now})`
												}

												console.log('matches_condition',matches_condition);

												console.log('type',type);

												try{
													var { results } = await env[`commerce_logis_${zoneRegion}_${type}`].prepare(`SELECT * FROM ${type} WHERE ${matches_condition} LIMIT 100`).all()

													if(results.length){
														for(var r = 0; r < results.length; r++){
															var result = results[r]

															var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(result.data))

															var data = JSON.parse(decompressedJsonString)

															if(data){
																if(Object.keys(data).length){
																	for (var name in data) {
																		if (data.hasOwnProperty(name)) {
																			var value = data[name]

																			result[name] = value
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
												}catch(err){
													console.log('matches_condition',err);
												}
												
											}

											var orderBy = ''

											if(context.sort && context.by){
												orderBy = `ORDER BY ${context.by} ${context.sort}`
											}

											console.log('type',type);

											console.log('after condition',condition);

											if(condition){
												try{
													if(condition.indexOf(created_at) > -1){
														condition += ` AND "created_at" < ${now}`
													}

													console.log(`SELECT * FROM ${type} WHERE ${condition} AND "to" = '${team.id}' ${orderBy} LIMIT 300`);
													 
													var { results } = await env[`commerce_logis_${zoneRegion}_${type}`].prepare(`SELECT * FROM ${type} WHERE ${condition} AND "to" = '${team.id}' ${orderBy} LIMIT 300`).all()

													console.log('results.length',results.length);

													if(results.length){
														for(var r = 0; r < results.length; r++){
															var result = results[r]

															var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(safeClone(result.data)))

															var data = JSON.parse(decompressedJsonString)

															if(data){
																result.data = {}

																if(Object.keys(data).length){
																	for (var name in data) {
																		if (data.hasOwnProperty(name)) {
																			var value = data[name]

																			if(name == "type" || name == "item" || name == "node" || name == "link" || name == "origin" || name == "detail"){

																			}else{
																				result.data[name] = value
																			}
																		}
																	}
																}
															}

															results[r].data = result.data

															delete results[r].from
															delete results[r].to
															delete results[r].cc
															delete results[r].bcc
															delete results[r].ref
														}

														rag.search.sql = {
															results : results
														}
													}
												}catch(err){
													console.log('text2sql err',err);
												}

											}

												
												


											var system = 'Return the content related to the {search.text} value from the search results in a JSON structure.'

											var content = context2results(context, [...rag.search.sql.results, ...rag.search.vector.results], language)


											var generation

											if(models['deepinfra']){
												generation = await Deepinfra(deepinfra, 'openai/gpt-oss-20b', system, content)

												models['deepinfra'] -= 1

											}

											if(!generation && gemini_llm_api){
												generation = await Gemini(gemini_llm_api, gemini_llm_model, system, content)

												models[gemini_llm_api+'-'+gemini_llm_model] -= 1

											}

											if(!generation){
												fallback = 'overflow'

												continue
											}


											console.log('generation', JSON.stringify(generation));

											var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
												id : context.id,
												markdown : generation.markdown,
												results: generation.results,
												search : rag.search
											})), { to: 'arraybuffer' })

											context.data = arr.buffer

											statements[`commerce_logis_${zoneRegion}_talks`].push(
												env[`commerce_logis_${zoneRegion}_talks`].prepare(`
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
													task.id,
													context.type,
													team.id,
													task.bcc,
													task.cc,
													task.bcc,
													task.ref,
													context.data,
													now-100,
													now
												)
											)

											statements[`commerce_logis_${zoneRegion}_talks`].push(
												env[`commerce_logis_${zoneRegion}_talks`].prepare(`
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
													team.id,
													task.bcc,
													task.cc,
													task.bcc,
													task.ref,
													context.data,
													now,
													now
												)
											)
										}
									}
								}	
							}

							// for loop end
						}

						
							
						if(Object.keys(statements).length){
							if(fallback){
								console.log('fallback',fallback);

								statements[`commerce_logis_${zoneRegion}_talks`].push(
									env[`commerce_logis_${zoneRegion}_talks`].prepare(`
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

							statements[`commerce_logis_${zoneRegion}_talks`].push(
								env[`commerce_logis_${zoneRegion}_talks`].prepare(`
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

							// statements[`commerce_logis_${zoneRegion}_talks`].push(
							// 	env[`commerce_logis_${zoneRegion}_talks`].prepare(`
							// 		UPDATE talks SET updated_at = ? WHERE id = ?
							// 	`).bind(
							// 		now, task.id
							// 	)
							// )

							

							var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({id : team.id, flag : task.flag, team : team, created_at : cron.created_at })), { to: 'arraybuffer' })

							statements[region].push(
								env[region].prepare(`
									UPDATE tasks SET data = ?, updated_at = ? WHERE id = ?
								`).bind(
									arr.buffer, now, task.id
								)
							)

							console.log('team.data.base.pages',JSON.stringify(team.data.base.pages));

							var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify(team.data)), { to: 'arraybuffer' })

							statements[logisRegion].push(
								env[logisRegion].prepare(`
									UPDATE users SET data = ?, updated_at = ? WHERE id = ?
								`).bind(
									arr.buffer, now, team.id
								)
							)






							limits[task.id.toUpperCase()] = true
						}
					}
				}catch(err){
					console.log('err',err);
				}

				if(region){
					statements[region].push(
						env[region].prepare(`
							DELETE FROM tasks WHERE "id" = 'lock'
						`)
					)

					for (const region in statements) {
						if (statements.hasOwnProperty(region)) {
							var batch = statements[region]

							if(batch.length){
								var { results, success, error } = await env[region].batch(batch)
							}
						}
					}
				}	

				var results = {}

				if(limits && models){
					results = {
						models : models,
						limits : limits
					}
				}

				return new Response(JSON.stringify(results), {
					headers: { "Content-Type": "application/json" },
				})
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