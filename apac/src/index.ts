import { Node, parseHTML } from 'linkedom'

import { gzip, ungzip } from 'pako'

import { ethers } from "ethers"


/*
	--- 결제 타입 ---
		$user
		$team

		+++ 결제 플로우 만들어야함

	***selector 가 같은데 계속 풀 html 문서 전송막기
*/

/*

	*** 중요 ***
		1 리스트 페이지는 프롬프트로 무조건 처리해야함

		2 상세는 CSS selector로 발라내야함

		* 상세 먼저 크롤링할 경우 "리스트 먼저 크롤링하라고" 안내 메세지 띄우기

		* 스캔이라고 하지 않고 리스트 스캔이라고 하기




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


*/


function crc32(s) { var polynomial = arguments.length < 2 ? 0x04C11DB7 : arguments[1], initialValue = arguments.length < 3 ? 0xFFFFFFFF : arguments[2], finalXORValue = arguments.length < 4 ? 0xFFFFFFFF : arguments[3], crc = initialValue, table = [], i, j, c; function reverse(x, n) { var b = 0; while (n) { b = b * 2 + x % 2; x /= 2; x -= x % 1; n--; } return b; } for (i = 256; i >= 0; i--) { c = reverse(i, 32); for (j = 0; j < 8; j++) { c = ((c * 2) ^ (((c >>> 31) % 2) * polynomial)) >>> 0; } table[i] = reverse(c, 32); } for (i = 0; i < s.length; i++) { c = s.charCodeAt(i); if (c > 255) { throw new RangeError(); } j = (crc % 256) ^ c; crc = ((crc / 256) ^ table[j]) >>> 0; } return (crc ^ finalXORValue) >>> 0; }


const randomKey = function(){
	var key = Math.random().toString()

	return parseInt(key.replace("0.",""))
}

const image2json = function(type){
	if(type == "tracking"){
		return `convert the shipping label image to fit the dataset JSON structure. Return only the JSON structure result, no explanation.{
			type:"shipping label",
			status:"draft" or "progress" or "return" or "complete",
			id:tracking number | string,
			title:${type} goods title | string, 
			senderName:senderName | string,
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
			shipping_method:"standard" or "express" or "same_day" or "pick_up" or "freight",
			shipping_duration:Estimated delivery days | number,
			bundle_shipping:Allow combined shipping | string,
			shipping_date:yyyy-MM-dd'T'HH:mm:ss | string,
		}`
	}
}



const type2json = function(type){
	if(type == 'tracking'){
		return ` 
			status:"draft" or "progress" or "return" or "complete",
			id:tracking number | string,
			title:${type} goods title | string, 
			senderName:senderName | string,
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
			shipping_method:"standard" or "express" or "same_day" or "pick_up" or "freight",
			shipping_duration:Estimated delivery days | number,
			bundle_shipping:Allow combined shipping | string,
			shipping_date:yyyy-MM-dd'T'HH:mm:ss | string,
		`
	}else if(type == 'sales'){
		return `
			id:Refer to the ID value from the link or an attribute | string,
			status:'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or "complete",
			payment_method:payment method | string,
			bank:bank company name | string,
			card:card company name | string,
			code:product constant code | string,
			model_name:product Model name | string,
			brand_name:product Brand name | string,
			condition:["new" or "used" or "lease" or "rental" or "refurbish"],
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
			cost_price:product cost price | number,
			compare_at_price:product Original price for showing discounts | number,
			stock_quantity:product Inventory quantity | number,
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
			shipping_method:"standard" or "express" or "same_day" or "pick_up" or "freight",
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
					link:URL includes the path additional goods link | string
				}
			],
			title:product based title | string,
			link:product detail link | string,
			date:yyyy-MM-dd'T'HH:mm:ss | string,
		`
	}else if(type == 'order'){
		return `
			status:'draft' or 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or "complete",
			order_goods:[{
				title:goods title | string,
				options:[{
					name : goods option name | string,
					option:goods option value | string,
				}],
				link:URL includes the path additional goods item link | string,
				id:Refer to the ID value from the link or an attribute | string,
			}],
			payment_date:payment_date | string,
			payment_method:'C.O.D.' or 'CARD' or 'BANK' or '',
			payment_origin:payment origin | string,
			date:yyyy-MM-dd'T'HH:mm:ss | string
		`
	}else if(type == 'coupon' || type == 'event'){
		return `
			type:'percentage' or 'fixed_amount' or 'free_shipping' or '',
			status:'draft' or 'progress' or 'stop' or 'cancel' or 'expire' or "complete",
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
			status:'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or "complete"
			name:${type} name | string,
			title:${type} item title | string, 
			completed:order complete | boolean,
			created_at:yyyy-MM-dd'T'HH:mm:ss
		`
	}
}

const context2intents = function(language){
	return `Return the intent from the sentence as a JSON object. {
		language:'${language}',
		type:['sales' or 'order' or 'goods' or 'tracking' or 'search' or 'review' or 'member' or 'coupon' or 'event' or ''],
		find:'many' or 'few' or 'much' or 'little' or '',
		criteria:['width' or 'height' or 'length' or 'weight' or 'shipping_fee' or 'shipping_duration' or 'sale_price' or 'cost_price' or 'stock_quantity' or 'low_stock_threshold' or 'discount' or 'min_order_amount' or 'max_discount_amount' or 'usage_limit' or 'usage_per' or 'started_at' or 'expired_at'],
		status'draft' or 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or "complete",
	}`
}

	// convert the natural language content to fit the dataset JSON structure. Return only the JSON structure result, no explanation. 
	// {
	// 	filters:{
	// 		quantity:{
	// 			eq,lte,gte:0,
	// 		},
	// 		amount:{
	// 			currency:"",
	// 			eq,lte,gte:0,
	// 		},
	// 		date:{
	// 			eq:"${current}",lte:"${current}",gte:"${current}"
	// 		},
	// 		${type2json(prompt.type)}
	// 	},
	// 	text:translate the semantic content related to 'type' into English, excluding any mention of 'filters', excluding any mention of 'find'
	// }

const text2json = function(language, prompt, range, current){
	var width = "";
	var height = "";
	var length = "";
	var weight = "";
	var shipping_fee = "";
	var shipping_duration = "";
	var sale_price = "";
	var cost_price = "";
	var stock_quantity = "";
	var low_stock_threshold = "";
	var discount = "";
	var min_order_amount = "";
	var max_discount_amount = "";
	var usage_limit = "";
	var usage_per = "";

/*

convert the natural language content to fit the dataset JSON structure.
{
	sql : {
		where : [
			{
				type:'sales' or 'order' or 'goods' or 'tracking' or 'search' or 'view' or 'review' or 'member' or 'coupon' or 'event' or '',
				find:'many' or 'few' or 'much' or 'little' or '',
				status'draft' or 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or "complete",
				intent:intent,
				condition : {
					quantity:{
						eq:0,lte:0,gte:0,
					},
					amount:{
						currency:"",
						eq:0,lte:0,gte:0,
					},
				},
				orderBy:'size' or 'weight' or 'shipping_fee' or 'shipping_duration' or 'sale_price' or 'cost_price' or 'low_stock_threshold' or 'discount' or 'min_order_amount' or 'max_discount_amount' or 'usage_limit' or 'usage_per' or '',
				text:translate the semantic content related to 'type' into English, excluding any mention of 'condition', excluding any mention of 'find'
			},
		]
	}
}
'여름 시즌' 기획전에 포함된 상품들 중, 상세 페이지 조회수는 상위 20%에 속하지만 구매 전환율이 1% 미만인 상품들만 따로 보여줘. 원인 분석이 시급해

*/


	return `convert the natural language content to fit the dataset JSON structure.
	- The time value is answered based on "${current}"
	{
		sql: {
			where : [
				{
					type:'sales' or 'order' or 'goods' or 'tracking' or 'search' or 'review' or 'member' or 'coupon' or 'event' or '',
					find:'many' or 'few' or 'much' or 'little' or '',
					status:'draft' or 'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or "complete",
					intent:intent,
					entity:entity,
					condition:{
						date:{
							eq:"${current}",lte:"${current}",gte:"${current}"
						},
						quantity:{
							eq:0,lte:0,gte:0,
						},
						amount:{
							currency:"",
							eq:0,lte:0,gte:0,
						},
					},
					orderBy:'size' or 'weight' or 'shipping_fee' or 'shipping_duration' or 'sale_price' or 'cost_price' or 'low_stock_threshold' or 'discount' or 'min_order_amount' or 'max_discount_amount' or 'coupon_usage_limit' or 'coupon_usage_per_customer' or '',
					text:translate the semantic content related to 'type' into ${language}, excluding any mention of 'condition', excluding any mention of 'find'
				},
			]
		}
	}`
}




const list2json = function(language){
	return `
		type:'order' or 'goods' or 'tracking' or 'search' or 'review' or 'member' or 'coupon' or 'event' or '',
		list:item parent list CSS selector excluding ads,
		item:Item CSS selector excluding ads,
		more:item detail link CSS selector,
		next:items next button CSS selector,
		text:Summarize the contents of the items array in ${language},
		items: [
			if (type is 'tracking' or 'review' or 'member') {
				status:'start' or 'progress' or 'stop' or 'cancel' or 'return',
				id:Refer to the ID value from the link or an attribute | string,
				title:author and content | string, 
				date:yyyy-MM-dd'T'HH:mm:ss | string,
			}
			if (type is 'order' or 'goods') {
				status:'active' or 'progress' or 'remove' or 'hide' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or "complete",
				id:Refer to the ID value from the link or an attribute | string,
				title:title | string, 
				sale_price:sale price | number,
				link:detail link | string,
				currency:type based item currency | string,
				stock_quantity:item stock quantity | number,
				date:yyyy-MM-dd'T'HH:mm:ss | string,
			}
			if (type is 'coupon' or 'event') {
				status : 'active' or 'progress' or 'hide' or 'stop' or 'cancel' or 'expire' or "complete",
				id:Refer to the ID value from the link or an attribute | string,
				title:type based item title, 
				started_at:yyyy-MM-dd'T'HH:mm:ss,
				expired_at:yyyy-MM-dd'T'HH:mm:ss,
			}
		] 
	}`
}


const item2json = function(type){
	if(type == 'tracking' || type == 'review' || type == 'member'){
		return `
			list:${type} item parent list CSS selector excluding ads,
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
			cost_price:Cost price CSS selector,
			compare_at_price:Original price for showing discounts CSS selector,
			stock_quantity:Inventory quantity CSS selector,
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
				link:URL includes the path additional goods link CSS selector
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
				link:URL includes the path additional product link CSS selector
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
// 	if (!elements || elements.length === 0) {
// 		return null;
// 	}

// 	// Start with the first element's parent as the potential common ancestor
// 	let ancestor = elements[0].parentNode;

// 	// Loop through all elements
// 	for (let i = 1; i < elements.length; i++) {
// 		// Check if the current ancestor contains the next element
// 		// If not, move up the tree from the first element
// 		if (!ancestor.contains(elements[i])) {
// 			ancestor = ancestor.parentNode;
// 			// Restart the loop to re-check all elements with the new ancestor
// 			i = 0; 
// 		}
// 	}

// 	return ancestor;
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
	const indent = '  '.repeat(indentLevel); // 들여쓰기 문자 (공백 2칸)
	let lines = [];

	nodes.forEach(node => {
		// 1. Element 노드 처리
		if (node.nodeType === Node.ELEMENT_NODE) {
			const tagName = node.tagName.toLowerCase();

			// --- ✨ 추가된 부분: base64 이미지를 포함하는 img 태그 제외 ---
			const src = node.getAttribute('src');
			if (tagName === 'img' && src && src.includes('base64')) {
				return; // src에 'base64'가 포함된 img 태그는 변환에서 건너뜁니다.
			}
			// --- 제외 로직 끝 ---

			// 불필요한 태그들을 만나면 건너뛰기
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

			// <img> 태그의 src 속성 처리
			if (tagName === 'img' && node.hasAttribute('src')) {
				const src = node.getAttribute('src');
				if (src) { // src 속성값이 비어있지 않은 경우에만 추가
					otherAttributes.push(`src="${src}"`);
				}
			}

			// <a> 태그의 href 속성 처리
			if (tagName === 'a' && node.hasAttribute('href')) {
				const href = node.getAttribute('href');
				if (href) { // href 속성값이 비어있지 않은 경우에만 추가
					otherAttributes.push(`href="${href}"`);
				}
			}

			// ✨ 추가된 부분: data- 속성 처리
			// NamedNodeMap을 Array로 변환하여 모든 속성을 순회합니다.
			Array.from(node.attributes).forEach(attr => {
				if (attr.name.startsWith('data-')) {
					otherAttributes.push(`${attr.name}="${attr.value}"`);
				}
			});
			// ✨ 추가된 부분 끝

			// 괄호로 묶는 속성들 추가 (src="..." href="...")
			if (otherAttributes.length > 0) {
				attributesString += `(${otherAttributes.join(' ')})`;
			}
			// --- 속성 처리 끝 ---


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

			if (currentNode.hasChildNodes()) {
				lines = lines.concat(generatePugLines(currentNode.childNodes, indentLevel + 1));
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


const twoPartDomains = ["co.kr","co.uk","co.jp","com.cn","co.in","com.mx","co.id","com.my","com.sg","com.ph","com.vn"];


// 국가 코드를 지역으로 매핑하는 맵
// 국가 코드를 지역으로 매핑하는 맵 (ISO 3166-1 alpha-2 기준)

/*
	logis 
		- pages 
		- tasks

	사용자 5000명씩 분할
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



/*
	before 가져올것
	after 기준값
	item after item


	추후 벡터 db 검색시 내용이 많아지면 토큰 소모가 커질수 있으므로 distinct 꼭 사용하기
*/
const Flow = function(query, item){
	if(query == "goods" && item.type == "order"){
		return {
			type : 'sales',
			column : 'index',
			index : item.sales
		}

	}else if(query == "tracking" && item.type == "order"){
		return {
			type : 'tracking',
			column : 'index',
			index : item.tracking
		}

	}else if(query == "coupon" && item.type == "order"){
		return {
			type : 'event',
			column : 'index',
			index : item.event
		}

	}else if(query == "event" && item.type == "order"){
		return {
			type : 'event',
			column : 'index',
			index : item.event
		}




	}else if(query == "order" && item.type == "goods"){
		return {
			type : 'sales',
			column : 'sales',
			index : item.id
		}
		
	}else if(query == "tracking" && item.type == "goods"){
	// 	return {
	// 		type : 'sales',
	// 		column : 'sales',
	// 		index : item.index,
	// 		flow : {
	// 			type : 'tracking',
	// 			column : 'index',
	// 			index : item.index
	// 		}
	// 	}

	}else if(query == "event" && item.type == "goods"){
	// 	return {
	// 		type : 'event',
	// 		column : 'index'
	// 	}

	}else if(query == "coupon" && item.type == "goods"){
	// 	return {
	// 		type : 'event',
	// 		column : 'index'
	// 	}




	}else if(query == "goods" && item.type == "tracking"){
		return {
			type : 'sales',
			column : 'tracking',
			index : item.index
		}

	}else if(query == "order" && item.type == "tracking"){
		return {
			type : 'sales',
			column : 'tracking',
			index : item.index
		}

	}else if(query == "event" && item.type == "tracking"){
		// return {
		// 	type : 'sales',
		// 	column : 'tracking',
		// 	index : item.index
		// 	flow : {
		// 		type : 'event',
		// 		column : 'index',
		// 		index : 'event'
		// 	}
		// }

	}else if(query == "coupon" && item.type == "tracking"){
		// return {
		// 	type : 'sales',
		// 	column : 'tracking',
		// 	index : item.index
		// 	flow : {
		// 		type : 'event',
		// 		column : 'index',
		// 		index : 'event'
		// 	}
		// }




	}else if(query == "sales" && item.type == "event"){
		return {
			type : 'sales',
			column : 'event',
			index : item.index
		}

	}else if(query == "order" && item.type == "event"){
		return {
			type : 'sales',
			column : 'event',
			index : item.index
		}

	}else if(query == "tracking" && item.type == "event"){
		return {
			type : 'sales',
			column : 'event',
			index : item.index,
			flow : {
				type : 'tracking',
				column : 'index',
				index : 'tracking'
			}
		}

	}else if(query == "coupon" && item.type == "event"){
		return {
			type : 'event',
			column : 'index',
			index : item.index
		}




	}else if(query == "goods" && item.type == "coupon"){
		return {
			type : 'sales',
			column : 'event'
		}

	}else if(query == "order" && item.type == "coupon"){
	// 	return {
	// 		type : 'sales',
	// 		column : 'event'
	// 	}

	}else if(query == "tracking" && item.type == "coupon"){
		// return {
		// 	type : 'sales',
		// 	column : 'event',
		// 	flow : {
		// 		type : 'tracking',
		// 		column : 'index',
		// 		index : 'event'
		// 	}
		// }

	}else if(query == "event" && item.type == "coupon"){
		return {
			type : 'event',
			column : 'event',
			index : item.index
		}

	}


	return {
		type : null,
		column : null
	}
}


/*
	벡터맵으로 구분하자
	wnam-logis		Western North America
	enam-logis		Eastern North America
	weur-logis		Western Europe
	eeur-logis		Eastern Europe
	apac-logis		Asia-Pacific
	oc-logis			Oceania


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
	'eg': 'Arabic',  // Egypt

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

async function Deepinfra(key, model, system, user){
	// DeepInfra API 호출
	var body = {
		"model" : model,
		"messages": [
			{ "role": "system", "content": system },
			{ "role": "user", "content": user }
		],
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

	const res = await fetch(`https://api.deepinfra.com/v1/openai/${pathname}`, {
		method: "POST",
		headers: {
			"Authorization": `Bearer ${key}`,
			"Content-Type": "application/json",
		},
		body: JSON.stringify(body),
	});

	const json = await res.json();

	if(isEmbedding){
		return json.data[0].embedding
	}else{
		var content = json.choices[0].message.content;

		return content
	}
}

async function Gemini(key, model, system, user, config, inlineData){
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

	const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${key}`, {
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

	const data = await res.json()

	var content = data.candidates[0].content.parts[0].text

	if(config["response_mime_type"]){
		try{
			var results = JSON.parse(content)

			return results.length ? results[0] : results
		}catch(err){
			
		}
	}

	return content
}


async function Cron(event, env, ctx, models, limits){
	/*
		매월 1일에 결제한 사용자를 기준으로 
		사용 가능한 balance 지급하는 프로세스 추가해야함
	*/
	var now = Date.now()
	
	var created_at = now - 10000

	try{
		var { results } = await env[env.region].prepare(`SELECT * FROM tasks WHERE "created_at" < ${created_at} AND "updated_at" = 0 ORDER BY created_at ASC LIMIT 1000`).all()

		var len = results.length

		console.log('tasks len',len)

		var tasks = []

		var clear_condition = ""

		if (len) {
			for(var i = 0; i < len; i++){
				var cron = results[i]

				var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(cron.task))

				var task = JSON.parse(decompressedJsonString)

				if(task.method){
					delete task.method
				}

				tasks.push(task)
			}

			var pageCount = {}

			// tasks DB 저장하는것 추가해야함
			if(tasks.length){
				for(var t = 0; t < tasks.length; t++){
					var task = tasks[t]


					var geminiKey = function(gemini1, gemini2){
						if(Math.floor(Math.random() * 2)){
							return {
								first :gemini1,
								second:gemini2
							}
						}else{
							return {
								first :gemini2,
								second:gemini1
							}
						}
					}

					var geminiModel = function(){
						if(Math.floor(Math.random() * 2)){
							return {
								first :'gemini-2.5-flash-lite',
								second:'gemini-2.0-flash-lite'
							}
						}else{
							return {
								first :'gemini-2.0-flash-lite',
								second:'gemini-2.5-flash-lite'
							}
						}
					}



					var gemini_key = geminiKey(env.gemini1, env.gemini2)

					var gemini_model = geminiModel()

					var gemini_llm_api = ""

					var gemini_llm_model = ""

					if(models[`${gemini_key.first}-${gemini_model.first}`]){
						gemini_llm_api = gemini_key.first
						gemini_llm_model = gemini_model.first

						models[`${gemini_key.first}-${gemini_model.second}`]

					}else if(models[`${gemini_key.first}-${gemini_model.second}`]){
						gemini_llm_api = gemini_key.first
						gemini_llm_model = gemini_model.second

					}else if(models[`${gemini_key.second}-${gemini_model.first}`]){
						gemini_llm_api = gemini_key.second
						gemini_llm_model = gemini_model.first

					}else if(models[`${gemini_key.second}-${gemini_model.second}`]){
						gemini_llm_api = gemini_key.second
						gemini_llm_model = gemini_model.second

					}else{
						clear_condition += ` AND "id" != "${task.id}"`

						continue
					}

					var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
						now : now,
						ref : task.ref,
						region : env.region,
						models : models,
						limits : limits,
						counts : pageCount,
						gemini_llm_api : gemini_llm_api,
						gemini_llm_model : gemini_llm_model,
						deepinfra : env.deepinfra
					})), { to: 'arraybuffer' })


					const response = await fetch(`https://proxy.logis.center/`, {
						method: "POST",
						headers: {
							"Content-Type": "application/json"
						},
						body: arr.buffer
					});

					var success = response.clone();
					
					var fail = response.clone();

					try{
						var results = await success.json();

						models = results.models
						limits = results.limits

						pageCount = results.counts

					}catch(err){
						var text = await fail.text();
						await env[CenterRegion].prepare(`
							INSERT INTO console (
								"id", "bcc", "log", "created_at"
							) VALUES (
								?1, ?2, ?3, ?4
							) ON CONFLICT (id) DO NOTHING
						`).bind(
							hashId(),
							task.bcc,
							'task fetch'+text,
							now  // Parameter for created_at (only insert)
						).run()

					}
					
					// 우회 요청 종료 처리
				}
			}


		}
	}catch(err){
		console.log('batch err',err)
	}

	return {
		length:len,
		models:models,
		limits:limits
	}
}

export default {
	async scheduled(
		event: ScheduledEvent,
		env: Env,
		ctx: ExecutionContext
	): Promise<void> {
		var limits = {}
		var models = {}

		models['deepinfra'] = 10000
		models['cloudflare'] = 3000

		models[`${env.gemini1}-gemini-2.5-flash-lite`] = 4000
		models[`${env.gemini2}-gemini-2.5-flash-lite`] = 4000
		models[`${env.gemini1}-gemini-2.0-flash-lite`] = 4000
		models[`${env.gemini2}-gemini-2.0-flash-lite`] = 4000


		var started_at = performance.now()

		var expired_at = started_at + 60000

		var delay = 1

		while(true){
			var current_at = performance.now()

			if(expired_at < current_at){
				break
			}

			var results = await Cron(event, env, ctx, models, limits)

			limits = results.limits

			models = results.models

			if(results.length){
				delay = 1
			}else{
				delay += 1
			}

			await Sleep(3000 * delay)
		}
	}
} satisfies ExportedHandler<Env>