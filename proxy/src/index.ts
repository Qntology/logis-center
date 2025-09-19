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



export default {
	async fetch(
		request: Request,
		env: Env,
		ctx: ExecutionContext
	): Promise<Response> {
		// task 실행

		var headers = new Headers()

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

				console.log('json.ref',json.ref);

				var { results } = await env[region].prepare(`SELECT * FROM tasks WHERE "ref" = "${json.ref}" AND "created_at" < ${created_at} AND "updated_at" = 0 ORDER BY created_at ASC LIMIT 1`).all()

				console.log('results.length',results.length);

				if(results.length){
					for(var c = 0; c < results.length; c++){
						var cron = results[c]

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
							clear_condition += ` AND "id" != "${task.id}"`

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


						var page_count = team.data ? team.data.page_count : 0

						if(pageCount[task.bcc+task.cc]){
							page_count = pageCount[task.bcc+task.cc] = pageCount[task.bcc+task.cc] + page_count
						}else{
							page_count = pageCount[task.bcc+task.cc] = page_count
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


						// if(limits[task.payer]){
						// 	limits[task.payer] -= 1
						// }else{
						// 	clear_condition += ` AND "id" != "${task.id}"`

						// 	var item = {
						// 		id : hashId(task.id),
						// 		type : "cancel",
						// 		from : task.from,
						// 		to : task.to,
						// 		cc : task.cc,
						// 		bcc : task.bcc,
						// 		ref : task.id,
						// 		started_at : now,
						// 		updated_at : now
						// 	} 


						// 	statements[`${zoneRegion}_items`].push(
						// 		env[`${zoneRegion}_items`].prepare(`
						// 			INSERT INTO items (
						// 				"id", "type", "from", "to", "cc", "bcc", "ref", "created_at", "updated_at"" 
						// 			) VALUES (
						// 				?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9
						// 			) ON CONFLICT (id) DO UPDATE SET
						// 				"type" = EXCLUDED."type",
						// 				"from" = EXCLUDED."from",
						// 				"to" = EXCLUDED."to",
						// 				"cc" = EXCLUDED."cc",
						// 				"bcc" = EXCLUDED."bcc",
						// 				"ref" = EXCLUDED."ref",
						// 				"created_at" = EXCLUDED."created_at",
						// 				"updated_at" = EXCLUDED."updated_at"
						// 		`).bind(
						// 			item.id,
						// 			item.type,
						// 			item.from,
						// 			item.to,
						// 			item.cc,
						// 			item.bcc,
						// 			item.ref,
						// 			now,
						// 			now
						// 		)
						// 	)

						// 	continue
						// }


						// model context protocol

						var prefix = type2json(task.type)
						
						if(task.contentType == "image/jpeg"){
							var base64 = arrayBufferToBase64(task.buffer)

							var inlineData = { mimeType: task.contentType, data: base64 }

							var type = talk.type = task.type

							var system = image2json(type)

							var content = task.text

							var item = await Gemini(gemini_llm_api, gemini_llm_model, system, content, null, inlineData)

							if(!item.status){
								// 올바르지 않은 이미지 안내하기

								continue
							}

							var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify(item)), { to: 'arraybuffer' })

							item.data = arr.buffer


							

							item.no = item.id + ""

							item.id = hashId(task.to+item.no)

							item.type = type

							item.from = task.from

							item.to = task.to

							item.cc = task.cc // logis.center로 잡혀져 있음 

							item.bcc = task.bcc

							item.ref = task.id

							item.created_at = now

							item.index = crc32(task.to+item.no)

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


							var system = semantic_prompt_system(language)

							if(models['deepinfra']){
								talk.text = await Deepinfra(env.deepinfra, 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', system, JSON.stringify(content))

								models['deepinfra'] -= 1

							}else if(gemini_llm_api){
								talk.text = await Gemini(gemini_llm_api, gemini_llm_model, system, JSON.stringify(content))

								models[gemini_llm_api+'-'+gemini_llm_model] -= 1

							}else{
								clear_condition += ` AND "id" != "${task.id}"`

								continue
							}


							var metadata = {
								id: item.id,
								type: item.type,
								from: task.from,
								to: task.to,
								cc: task.cc,
								bcc: task.bcc,
								ref:task.id
							}

							if(models['cloudflare']){
								var { data: embeddings } = await env.AI.run('@cf/baai/bge-m3', {
									text: [talk.text]
								})

								var $VectorizeVector = [
									{
										id: item.id,
										values: embeddings[0],
										metadata: metadata
									}
								]

								models['cloudflare'] -= 1

							}else if(models['deepinfra']){
								var embeddings = await Deepinfra(env.deepinfra, 'BAAI/bge-m3', '', talk.text)

								var $VectorizeVector: VectorizeVector[] = embeddings.map((values, i) => {
									return {
										id: item.id,
										values: values,
										metadata: metadata
									}
								})

								models['deepinfra'] -= 1

							}else{
								clear_condition += ` AND "id" != "${task.id}"`

								continue
							}

							await env[`${vectorRegion}-${type}`].upsert($VectorizeVector)



							var { results } = await env[`${zoneRegion}_${type}`].prepare(`SELECT * FROM ${type} WHERE "id" = "${item.id}" AND "to" = "${task.to}" AND "created_at" < ${now} LIMIT 1`).all()

							if(type == "tracking"){
								if(results.length){
									var tracking = results[0]

									statements[`${zoneRegion}_tracking`].push(
										env[`${zoneRegion}_tracking`].prepare(`
											UPDATE tracking SET ref = ?, updated_at = ?, status = ? WHERE id = ?
										`).bind(
											task.id, now, item.status, tracking.id
										)
									)
								}

								var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
									text : talk.text,
									title : item.title
								})), { to: 'arraybuffer' })

								item.data = arr.buffer

								statements[`${zoneRegion}_tracking`].push(
									env[`${zoneRegion}_tracking`].prepare(`
										INSERT INTO tracking (
											"id", "from", "to", "cc", "bcc", "ref", "created_at", "index", "status", "no", "sender_address", "sender_phone", "recipient_address", "recipient_phone", "width", "height", "length", "weight", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "shipping_date", "delivery_date", "order_date", "payment_date", "payment_method", "payment_origin", "payment_number", "bundle_shipping"
										) VALUES (
											?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30
										) ON CONFLICT (id) DO UPDATE SET
											"from" = EXCLUDED."from",
											"to" = EXCLUDED."to",
											"cc" = EXCLUDED."cc",
											"bcc" = EXCLUDED."bcc",
											"ref" = EXCLUDED."ref",
											"created_at" = EXCLUDED."created_at",
											"index" = EXCLUDED."index",
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
										item.from,
										item.to,
										item.cc,
										item.bcc,
										item.ref,
										item.created_at,
										item.index,
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

							statements[`${zoneRegion}_items`].push(
								env[`${zoneRegion}_items`].prepare(`
									INSERT INTO items (
										"id", "type", "from", "to", "cc", "bcc", "ref", "created_at", "updated_at"
									) VALUES (
										?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9
									) ON CONFLICT (id) DO UPDATE SET
										"type" = EXCLUDED."type",
										"from" = EXCLUDED."from",
										"to" = EXCLUDED."to",
										"cc" = EXCLUDED."cc",
										"bcc" = EXCLUDED."bcc",
										"ref" = EXCLUDED."ref",
										"created_at" = EXCLUDED."created_at",
										"updated_at" = EXCLUDED."updated_at"
								`).bind(
									hashId(task.id),
									"prompt",
									task.from,
									task.to,
									task.cc,
									task.bcc,
									task.id,
									now,
									0
								)
							)

						}else if(prefix){
							try{
								var system = `Analyze the provided Pug template and return it in the following JSON format, no explanation. {language:'${language}',${prefix.trim()} } `

								var content = convertHtmlToCleanPug(task.text)

								if(models['deepinfra']){
									var page = await Deepinfra(env.deepinfra, 'openai/gpt-oss-20b', system, content)

									models['deepinfra'] -= 1

								}else if(gemini_llm_api){
									var page = await Gemini(gemini_llm_api, gemini_llm_model, system, content)

									models[gemini_llm_api+'-'+gemini_llm_model] -= 1

								}else{
									clear_condition += ` AND "id" != "${task.id}"`

									continue
								}

							}catch(err){
								console.log('prefix err',err)
							}

						}else if(task.scan){
							// INSERT 백터 생성 INSERT

							var isDetail = false

							var page

							try{
								var system = list2json(language)

								if(task.referrer){
									var { results } = await env[CenterRegion].prepare(`SELECT * FROM pages WHERE "id" = "${task.referrer}" AND "cc" = "${task.cc}" AND "created_at" < ${created_at} LIMIT 1`).all()

									if(results.length){
										var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(results[0].data))

										var data = JSON.parse(decompressedJsonString)

										system = item2json(data.type)

										isDetail = true
									}
								}

								system = system.trim()


								var pageId = hashId(task.cc+task.link)

								var { results } = await env[CenterRegion].prepare(`SELECT * FROM pages WHERE "id" = "${pageId}" AND "cc" = "${task.cc}" AND "created_at" < ${created_at} LIMIT 1`).all()

								if(results.length){
									var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(results[0].data))

									var selectors = JSON.parse(decompressedJsonString)

									// 있는지 여부 확인해야함

									try{
										var { document } = parseHTML(`<html><body>${task.text}</body></html>`);

										if(selectors.type){
											var items = []

											for (const s in selectors) {
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

													items.push(item)
												}
											}

											if(items.length){
												page = selectors
												page.items = items
											}
										}

									}catch(err){
										console.log('page err',err);
									}
								}



								var content = convertHtmlToCleanPug(task.text)

								if(!page){
									if(gemini_llm_api){

										var page = await Gemini(gemini_llm_api, gemini_llm_model, `Analyze the provided Pug template and return it in the following JSON format, no explanation. {language:'${language}',${system}}`, content)

										models[gemini_llm_api+'-'+gemini_llm_model] -= 1

									}else{
										clear_condition += ` AND "id" != "${task.id}"`

										continue
									}

									// if(models['deepinfra']){
									// 	var page = await Deepinfra(env.deepinfra, 'openai/gpt-oss-20b', `Analyze the provided Pug template and return it in the following JSON format, no explanation. {language:'${language}',${system}}`, content)

									// 	page = JSON.parse(page)


									// 	models['deepinfra'] -= 1

									// }else if(gemini_llm_api){

									// 	var page = await Gemini(gemini_llm_api, gemini_llm_model, `Analyze the provided Pug template and return it in the following JSON format, no explanation. {language:'${language}',${system}}`, content)


									// 	models[gemini_llm_api+'-'+gemini_llm_model] -= 1

									// }else{
									// 	clear_condition += ` AND "id" != "${task.id}"`

									// 	continue
									// }

									page.id = hashId(task.cc+task.link)

									page.from = task.from
									page.to = task.to
									page.cc = task.cc
									page.bcc = task.bcc
								}


								talk.text = page.text

								// await env[CenterRegion].prepare(`
								// 	INSERT INTO console (
								// 		"id", "bcc", "log", "created_at"
								// 	) VALUES (
								// 		?1, ?2, ?3, ?4
								// 	) ON CONFLICT (id) DO NOTHING
								// `).bind(
								// 	hashId(),
								// 	task.bcc,
								// 	'talk.text'+talk.text,
								// 	now  // Parameter for created_at (only insert)
								// ).run()

								// await env[CenterRegion].prepare(`
								// 	INSERT INTO console (
								// 		"id", "bcc", "log", "created_at"
								// 	) VALUES (
								// 		?1, ?2, ?3, ?4
								// 	) ON CONFLICT (id) DO NOTHING
								// `).bind(
								// 	hashId(),
								// 	task.bcc,
								// 	'page.type'+page.type,
								// 	now  // Parameter for created_at (only insert)
								// ).run()


								// await env[CenterRegion].prepare(`
								// 	INSERT INTO console (
								// 		"id", "bcc", "log", "created_at"
								// 	) VALUES (
								// 		?1, ?2, ?3, ?4
								// 	) ON CONFLICT (id) DO NOTHING
								// `).bind(
								// 	hashId(),
								// 	task.bcc,
								// 	'zoneRegion'+zoneRegion,
								// 	now  // Parameter for created_at (only insert)
								// ).run()


								var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
									text : page.text || "",
									list : page.list || "",
									item : page.item || "",
									more : page.more || "",
									next : page.next || ""
								})), { to: 'arraybuffer' })

								page.ref = task.id

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

								var items = page.items


								if(items?.length){
									/*
										주문이후의 절차는 주문번호로 매칭해야함

										type : tracking					// 배송추적
																		// "고객 주문"" or "자사 재고" 등으로 추상화 매칭

										type : order
											order 파생 정보는
											전체 주문 목록 스캔하여
											주문 아이템 링크 클릭시
											레퍼러 참조 이벤트 추적하여 기록
											이 부분 크롬 익스텐션에서 해야함


											이 부분은 무조건 유료만 가능하게

												벡터 쿼리로 미리 저장하고 
													type : order, semantic : cancel		// 주문취소
													type : order, semantic : exchange	// 교환
													type : order, semantic : return 	// 반품
													type : order, semantic : refund		// 환불

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

										if(!item.title){
											continue
										}

										item.type = page.type

										item.no = (item.id ? item.id : i).toString()

										item.index = crc32(task.to+item.no)

										try{
											var url = new URL(item.link)

											var cc = hashId(url.host+url.pathname)

											if(cc == task.cc){
												item.link = url.pathname + url.search
											}
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

										if(item.link){
											if(item.link == task.link){
												item.id = hashId(task.id)
											}else{
												item.id = hashId(task.to+task.cc+item.link)
											}
										}else{
											item.id = hashId(task.to+task.cc+task.link+item.no)

											// tracking은 no 번호가 밀어내기가 될수 있어서 덮어씌우기로 반영됨
										}





										item.flag = task.flag
										
										item.from = task.from
										item.to = task.to
										item.cc = task.cc
										item.bcc = task.bcc

										item.ref = task.id



										item.currency = item.currency ? item.currency : ""

										item.quantity = item.quantity ? parseInt(item.quantity) : 0

										item.created_at = now

										item.updated_at = now

										item.semantic = item.title

										item.started_at = item.manufacture_date ? item.manufacture_date : 0
										
										item.expired_at = item.expiration_date ? item.expiration_date : 0





										var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
											id : item.id,
											title : item.title,
											link : item.link
										})), { to: 'arraybuffer' })

										item.data = arr.buffer


										if(containsUnsupportedChars(item.title)){
											var obj = {
												type : item.type,
												status : item.status
											}

											/*
												추후 주소 일때 () 괄호 영역 제거해야 정확해짐
													예를 들어 건물이름 층수 있으면 부정확해짐
											*/ 

											obj[language] = item.title ? item.title : ""

											if(item.type == "tracking"){
												if(item.sender){
													obj.sender = item.sender
												}

												if(item.recipient){
													obj.recipient = item.recipient
												}
											}

											var content = JSON.stringify(obj)

											if(gemini_llm_api){
												item.semantic = await Gemini(gemini_llm_api, gemini_llm_model, semantic_prompt_system(language), content, {"temperature": 1})

												models[gemini_llm_api+'-'+gemini_llm_model] -= 1

											}else{
												clear_condition += ` AND "id" != "${task.id}"`

												continue
											}

											// if(models['deepinfra']){
											// 	item.semantic = await Deepinfra(env.deepinfra, 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', semantic_prompt_system(language), content)

											// 	models['deepinfra'] -= 1

											// }else if(gemini_llm_api){
											// 	item.semantic = await Gemini(gemini_llm_api, gemini_llm_model, semantic_prompt_system(language), content, {"temperature": 1})

											// 	models[gemini_llm_api+'-'+gemini_llm_model] -= 1

											// }else{
											// 	clear_condition += ` AND "id" != "${task.id}"`

											// 	continue
											// }
										}


										try{
											await env[CenterRegion].prepare(`
												INSERT INTO console (
													"id", "bcc", "log", "created_at"
												) VALUES (
													?1, ?2, ?3, ?4
												) ON CONFLICT (id) DO NOTHING
											`).bind(
												hashId(),
												task.bcc,
												'item.semantic'+item.semantic,
												now  // Parameter for created_at (only insert)
											).run()
										}catch(err){

										}


										if(item.semantic){
											var metadata = {
												type: item.type,
												from: item.from,
												to: item.to,
												cc: item.cc,
												bcc: item.bcc,
												ref:item.ref
											}

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

											}else if(models['deepinfra']){
												var embeddings = await Deepinfra(env.deepinfra, 'BAAI/bge-m3', '', item.semantic)

												var $VectorizeVector: VectorizeVector[] = embeddings.map((values, i) => {
													return {
														id: item.id,
														values: values,
														metadata: metadata
													}
												})

												models['deepinfra'] -= 1

											}else{
												clear_condition += ` AND "id" != "${task.id}"`

												continue
											}

											await env[`${vectorRegion}-${itemType}`].upsert($VectorizeVector)
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



										if(itemType == "sales"){
											statements[`${zoneRegion}_sales`].push(
												env[`${zoneRegion}_sales`].prepare(`
													INSERT INTO sales (
														"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "started_at", "expired_at", "index", "event", "views", "sales", "width", "height", "length", "weight", "size", "currency", "cost_price", "sale_price", "discount", "quantity", "tracking", "phone", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "fulfillment_service", "stock_keeping_unit", "bundle_shipping", "used", "lease", "rental", "refurbish", "tax_included", "release_date"
													) VALUES (
														?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31, ?32, ?33, ?34, ?35, ?36, ?37, ?38, ?39, ?40
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
														"sales" = EXCLUDED."sales",
														"width" = EXCLUDED."width",
														"height" = EXCLUDED."height",
														"length" = EXCLUDED."length",
														"weight" = EXCLUDED."weight",
														"size" = EXCLUDED."size",
														"currency" = EXCLUDED."currency",
														"cost_price" = EXCLUDED."cost_price",
														"sale_price" = EXCLUDED."sale_price",
														"discount" = EXCLUDED."discount",
														"quantity" = EXCLUDED."quantity",
														"tracking" = EXCLUDED."tracking",
														"phone" = EXCLUDED."phone",
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
													item.started_at ? parseFloat(item.started_at) : 0,
													item.expired_at ? parseFloat(item.expired_at) : 0,
													item.index ? parseFloat(item.index) : 0,
													item.event ? parseFloat(item.event) : 0,
													item.views ? parseFloat(item.views) : 0,
													item.sales ? parseFloat(item.sales) : 0,
													item.width ? parseFloat(item.width) : 0,
													item.height ? parseFloat(item.height) : 0,
													item.length ? parseFloat(item.length) : 0,
													item.weight ? parseFloat(item.weight) : 0,
													item.size ? item.size : "",
													item.currency ? item.currency : "",
													item.cost_price? parseFloat(item.cost_price) : 0,
													item.sale_price? parseFloat(item.sale_price) : 0,
													item.discount ? parseFloat(item.discount) : 0,
													item.quantity ? parseFloat(item.quantity) : 0,
													item.tracking ? parseFloat(item.tracking) : 0,
													item.phone ? item.phone : "",
													item.carrier ? item.carrier : "",
													item.shipping_fee ? parseFloat(item.shipping_fee) : 0,
													item.shipping_method ? item.shipping_method : "",
													item.shipping_duration ? parseFloat(item.shipping_duration) : 0,
													item.fulfillment_service ? item.fulfillment_service : "",
													item.stock_keeping_unit ? item.stock_keeping_unit : "",
													item.bundle_shipping ? parseFloat(item.bundle_shipping) : 0,
													item.used ? parseFloat(item.used) : 0,
													item.lease ? parseFloat(item.lease) : 0,
													item.rental ? parseFloat(item.rental) : 0,
													item.refurbish ? parseFloat(item.refurbish) : 0,
													item.tax_included ? parseFloat(item.tax_included) : 0,
													item.release_date ? parseFloat(item.release_date) : 0
												)
											)
										}else if(itemType == "tracking"){
											statements[`${zoneRegion}_tracking`].push(
												env[`${zoneRegion}_tracking`].prepare(`
													INSERT INTO tracking (
														"id", "from", "to", "cc", "bcc", "ref", "created_at", "index", "status", "no", "sender_address", "sender_phone", "recipient_address", "recipient_phone", "width", "height", "length", "weight", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "shipping_date", "delivery_date", "order_date", "payment_date", "payment_method", "payment_origin", "payment_number", "bundle_shipping"
													) VALUES (
														?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30
													) ON CONFLICT (id) DO UPDATE SET
														"from" = EXCLUDED."from",
														"to" = EXCLUDED."to",
														"cc" = EXCLUDED."cc",
														"bcc" = EXCLUDED."bcc",
														"ref" = EXCLUDED."ref",
														"created_at" = EXCLUDED."created_at",
														"index" = EXCLUDED."index",
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
													item.from,
													item.to,
													item.cc,
													item.bcc,
													item.ref,
													item.created_at,
													item.index,
													item.status,
													item.no ? item.no : "",
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
										}else if(itemType == "event"){
											statements[`${zoneRegion}_event`].push(
												env[`${zoneRegion}_event`].prepare(`
													INSERT INTO event (
														"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "started_at", "expired_at", "index", "event", "phone", "address", "status", "code", "discount", "quantity", "usage_per", "usage_limit", "min_order_amount", "max_order_amount", "max_discount_amount", "new_customer_only", "first_purchase_only", "region_restrictions"
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
														"phone" = EXCLUDED."phone",
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
													item.started_at ? parseFloat(item.started_at) : 0,
													item.expired_at ? parseFloat(item.expired_at) : 0,
													item.index ? parseFloat(item.index) : 0,
													item.event ? parseFloat(item.event) : 0,
													item.phone ? item.phone : "",
													item.address ? item.address : "",
													item.status ? item.status : "",
													item.code ? item.code : "",
													item.discount ? parseFloat(item.discount) : 0,
													item.quantity ? parseFloat(item.quantity) : 0,
													item.usage_per ? parseFloat(item.usage_per) : 0,
													item.usage_limit ? parseFloat(item.usage_limit) : 0,
													item.min_order_amount ? parseFloat(item.min_order_amount) : 0,
													item.max_order_amount ? parseFloat(item.max_order_amount) : 0,
													item.max_discount_amount ? parseFloat(item.max_discount_amount) : 0,
													item.new_customer_only ? parseFloat(item.new_customer_only) : 0,
													item.first_purchase_only ? parseFloat(item.first_purchase_only) : 0,
													item.region_restrictions ? parseFloat(item.region_restrictions) : 0
												)
											)
										}



										var drafts = {}

										var related = Related(item.type)


										// 관련 타입 정보 가져옴

										for(var r = 0; r < related.length; r++){
											var relatedType = related[r]

											var { flow, type, column } = Flow(relatedType, item)

											// before ${type}에 ${column} index 값이 없으면 업데이트 해야함

											try{
												if(type){
													var { results } = await env[`${zoneRegion}_${type}`].prepare(
														`SELECT * FROM ${type} WHERE "${column}" = ? AND "to" = ? AND "cc" = ? AND "created_at" > ? LIMIT 1`
													).bind(
														crc32(task.to+item.id), team.id, item.cc, now - 60000
													).all()

													if(results.length){
														if(flow){
															var row = results[0]

															index = row[flow.index]

															var { results } = await env[`${zoneRegion}_${flow.type}`].prepare(
																`SELECT * FROM ${flow.type} WHERE "${flow.column}" = ? AND "to" = ? AND "cc" = ? AND "created_at" > ? LIMIT 1`
															).bind(
																index, team.id, item.cc, now - 60000
															).all()

															if(results.length){
																drafts[type] = {
																	rows : results,
																	flow : flow,
																	type : relatedType, 
																	column : column,
																	index : index
																}
															}
														}else{
															drafts[type] = {
																rows : results,
																type : relatedType, 
																column : column,
																index : index
															}
														}

													}else{
														// 없으면 추가해야함 - 일부 사용자가 직접 팝업으로 띄워야 할수 있음

														drafts[type] = { 
															rows : [],
															type : relatedType, 
															column : column,
															item : item
														}
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
													'inner err'+type+err,
													now  // Parameter for created_at (only insert)
												).run()
											}
										}



										if(Object.keys(drafts).length){
											for (const type in drafts) {
												if (drafts.hasOwnProperty(type)) {
													var draft = drafts[type]

													/*
														시나리오 case

														order 스캔 진행시
															draft.type == "goods" && row.type == "order"

															order items 만 있으면 goods 상세 정보가 없기 때문에 goods 정보 가져와서 order item에 업데이트 해야함


														goods 스캔 진행시 다음 의미 없음
															row.type == "goods"
															row.type == "event"
															row.type == "tracking"

														event 스캔 진행시 다음 의미 없음
															row.type == "coupon"
															row.type == "event"
															row.type == "goods"
															row.type == "order"

														tracking 스캔 진행시
															tracking 정보는 있고, order 정보에 tracking 값 업데이트 해야함
													*/

													if(draft.rows.length){
														// update만 진행

														// before ${type}에 ${column} index 값이 없으면 업데이트 해야함

														var column = draft.column

														var index = draft.index

														if(draft.flow){
															column = draft.flow.column
															index = draft.flow.index
														}

														for(var d = 0; d < draft.rows.length; d++){
															var row = draft.rows[d]

															var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(row.data))

															var data = JSON.parse(decompressedJsonString)

															if(draft.type == "goods" && row.type == "order"){
																var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
																	id : item.id,
																	title : item.title,
																	link : item.link,
																	data : data
																})), { to: 'arraybuffer' })

																row.data = arr.buffer

																statements[`${zoneRegion}_${type}`].push(
																	env[`${zoneRegion}_${type}`].prepare(`
																		INSERT INTO ${type} (
																			"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "started_at", "expired_at", "index", "event", "views", "sales", "width", "height", "length", "weight", "size", "currency", "cost_price", "sale_price", "discount", "quantity", "tracking", "phone", "carrier", "shipping_fee", "shipping_method", "shipping_duration", "fulfillment_service", "stock_keeping_unit", "bundle_shipping", "used", "lease", "rental", "refurbish", "tax_included", "release_date"
																		) VALUES (
																			?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31, ?32, ?33, ?34, ?35, ?36, ?37, ?38, ?39, ?40
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
																			"sales" = EXCLUDED."sales",
																			"width" = EXCLUDED."width",
																			"height" = EXCLUDED."height",
																			"length" = EXCLUDED."length",
																			"weight" = EXCLUDED."weight",
																			"size" = EXCLUDED."size",
																			"currency" = EXCLUDED."currency",
																			"cost_price" = EXCLUDED."cost_price",
																			"sale_price" = EXCLUDED."sale_price",
																			"discount" = EXCLUDED."discount",
																			"quantity" = EXCLUDED."quantity",
																			"tracking" = EXCLUDED."tracking",
																			"phone" = EXCLUDED."phone",
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
																		hashId(item.id+row.id),
																		item.type,
																		item.from,
																		item.to,
																		item.cc,
																		item.bcc,
																		item.ref,
																		row.data,
																		item.created_at,
																		row.started_at ? parseFloat(row.started_at) : 0,
																		row.expired_at ? parseFloat(row.expired_at) : 0,
																		item.index ? parseFloat(item.index) : 0,
																		row.event ? parseFloat(row.event) : 0,
																		item.views,
																		row.sales ? parseFloat(row.sales) : 0,
																		row.width ? parseFloat(row.width) : 0,
																		row.height ? parseFloat(row.height) : 0,
																		row.length ? parseFloat(row.length) : 0,
																		row.weight ? parseFloat(row.weight) : 0,
																		row.size ? row.size : "",
																		row.currency ? row.currency : "",
																		row.cost_price? parseFloat(row.cost_price) : 0,
																		row.sale_price? parseFloat(row.sale_price) : 0,
																		row.discount ? parseFloat(row.discount) : 0,
																		row.quantity ? parseFloat(row.quantity) : 0,
																		row.tracking ? parseFloat(row.tracking) : 0,
																		item.phone ? item.phone : "",
																		row.carrier ? row.carrier : "",
																		row.shipping_fee ? parseFloat(row.shipping_fee) : 0,
																		row.shipping_method ? row.shipping_method : "",
																		row.shipping_duration ? parseFloat(row.shipping_duration) : 0,
																		row.fulfillment_service ? row.fulfillment_service : "",
																		row.stock_keeping_unit ? row.stock_keeping_unit : "",
																		row.bundle_shipping ? parseFloat(row.bundle_shipping) : 0,
																		row.used ? parseFloat(row.used) : 0,
																		row.lease ? parseFloat(row.lease) : 0,
																		row.rental ? parseFloat(row.rental) : 0,
																		row.refurbish ? parseFloat(row.refurbish) : 0,
																		row.tax_included ? parseFloat(row.tax_included) : 0,
																		row.release_date ? parseFloat(row.release_date) : 0
																	)
																)


																var content = JSON.stringify({
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
																})


																if(models['deepinfra']){
																	var semantic = await Deepinfra(env.deepinfra, 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', semantic_prompt_system(language), content)

																	models['deepinfra'] -= 1

																}else if(gemini_llm_api){
																	var semantic = await Gemini(gemini_llm_api, gemini_llm_model, semantic_prompt_system(language), content, {"temperature": 1})

																	models[gemini_llm_api+'-'+gemini_llm_model] -= 1

																}else{
																	clear_condition += ` AND "id" != "${task.id}"`

																	continue
																}


																if(models['cloudflare']){
																	var { data: queryVector } = await env.AI.run('@cf/baai/bge-m3', {
																		text: [semantic],
																	})

																	models['cloudflare'] -= 1

																}else if(models['deepinfra']){
																	var queryVector = await Deepinfra(env.deepinfra, 'BAAI/bge-m3', '', semantic)

																	var $VectorizeVector: VectorizeVector[] = embeddings.map((values, i) => {
																		return {
																			id: item.id,
																			values: values,
																			metadata: metadata
																		}
																	})

																	models['deepinfra'] -= 1

																}else{
																	clear_condition += ` AND "id" != "${task.id}"`

																	continue
																}

															}else{
																// before ${type}에 ${column} index 값이 없으면 업데이트 해야함

																statements[`${zoneRegion}_${type}`].push(
																	env[`${zoneRegion}_${type}`].prepare(`
																		UPDATE ${type} SET ${column} = ? WHERE id = ?
																	`).bind(
																		index, row.id
																	)
																)

																statements[`${zoneRegion}_items`].push(
																	env[`${zoneRegion}_items`].prepare(`
																		UPDATE items SET updated_at = ? WHERE id = ?
																	`).bind(
																		now, row.id
																	)
																)
															}
														}
													}

												}
											}
													
										}
									}
								}



								task.no = page_count.toString()

								task.title = page.text // Analyze the provided Pug template and return it in the following JSON format

								task.semantic = page.text

								if(items?.length){
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

								try{
									statements[`${zoneRegion}_items`].push(
										env[`${zoneRegion}_items`].prepare(`
											INSERT INTO items (
												"id", "type", "from", "to", "cc", "bcc", "ref", "created_at", "updated_at"
											) VALUES (
												?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9
											) ON CONFLICT (id) DO UPDATE SET
												"type" = EXCLUDED."type",
												"from" = EXCLUDED."from",
												"to" = EXCLUDED."to",
												"cc" = EXCLUDED."cc",
												"bcc" = EXCLUDED."bcc",
												"ref" = EXCLUDED."ref",
												"created_at" = EXCLUDED."created_at",
												"updated_at" = EXCLUDED."updated_at"
										`).bind(
											hashId(task.id),
											task.type,
											task.from,
											task.to,
											task.cc,
											task.bcc,
											task.id,
											now,
											0
										)
									)
								}catch(err){
									console.log('item2 push err '+err)
								}

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
									now  // Parameter for created_at (only insert)
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


							var content = task.text


							// talk.type = prompt.type

							var range = {
								goods : {},
								order : {}
							}

							var base = {}

							if(team.data){
								base = team.data.base
							}



							if(base.range){
								if(Object.keys(base.range).length){
									
									base.range.price = {}

									if(base.range.price.min){
										range.price.min = `min:${base.range.price.min},`
									}

									if(base.range.price.max){
										range.price.max = `max:${base.range.price.max},`
									}

									

									base.range.quantity = {}

									if(base.range.quantity.min){
										range.quantity.min = `min:${base.range.quantity.min},`
									}

									if(base.range.quantity.max){
										range.quantity.max = `max:${base.range.quantity.max},`
									}



									base.range.width = {}

									if(base.range.width.min){
										range.width.min = `min:${base.range.width.min},`
									}

									if(base.range.width.max){
										range.width.max = `max:${base.range.width.max},`
									}



									base.range.height = {}

									if(base.range.height.min){
										range.height.min = `min:${base.range.height.min},`
									}

									if(base.range.height.max){
										range.height.max = `max:${base.range.height.max},`
									}



									base.range.length = {}

									if(base.range.length.min){
										range.length.min = `min:${base.range.length.min},`
									}

									if(base.range.length.max){
										range.length.max = `max:${base.range.length.max},`
									}



									base.range.weight = {}

									if(base.range.weight.min){
										range.weight.min = `min:${base.range.weight.min},`
									}

									if(base.range.weight.max){
										range.weight.max = `max:${base.range.weight.max},`
									}



									base.range.shipping_fee = {}

									if(base.range.shipping_fee.min){
										range.shipping_fee.min = `min:${base.range.shipping_fee.min},`
									}

									if(base.range.shipping_fee.max){
										range.shipping_fee.max = `max:${base.range.shipping_fee.max},`
									}



									base.range.shipping_duration = {}

									if(base.range.shipping_duration.min){
										range.shipping_duration.min = `min:${base.range.shipping_duration.min},`
									}
									
									if(base.range.shipping_duration.max){
										range.shipping_duration.max = `max:${base.range.shipping_duration.max},`
									}



									base.range.sale_price = {}

									if(base.range.sale_price.min){
										range.sale_price.min = `min:${base.range.sale_price.min},`
									}
									
									if(base.range.sale_price.max){
										range.sale_price.max = `max:${base.range.sale_price.max},`
									}



									base.range.cost_price = {}

									if(base.range.cost_price.min){
										range.cost_price.min = `min:${base.range.cost_price.min},`
									}
									
									if(base.range.cost_price.max){
										range.cost_price.max = `max:${base.range.cost_price.max},`
									}



									base.range.stock_quantity = {}

									if(base.range.stock_quantity.min){
										range.stock_quantity.min = `min:${base.range.stock_quantity.min},`
									}
									
									if(base.range.stock_quantity.max){
										range.stock_quantity.max = `max:${base.range.stock_quantity.max},`
									}



									base.range.low_stock_threshold = {}

									if(base.range.low_stock_threshold.min){
										range.low_stock_threshold.min = `min:${base.range.low_stock_threshold.min},`
									}
									
									if(base.range.low_stock_threshold.max){
										range.low_stock_threshold.max = `max:${base.range.low_stock_threshold.max},`
									}



									base.range.discount = {}

									if(base.range.discount.min){
										range.discount.min = `min:${base.range.discount.min},`
									}
									
									if(base.range.discount.max){
										range.discount.max = `max:${base.range.discount.max},`
									}



									base.range.min_order_amount = {}

									if(base.range.min_order_amount.min){
										range.min_order_amount.min = `min:${base.range.min_order_amount.min},`
									}
									
									if(base.range.min_order_amount.max){
										range.min_order_amount.max = `max:${base.range.min_order_amount.max},`
									}



									base.range.max_discount_amount = {}

									if(base.range.max_discount_amount.min){
										range.max_discount_amount.min = `min:${base.range.max_discount_amount.min},`
									}
									
									if(base.range.max_discount_amount.max){
										range.max_discount_amount.max = `max:${base.range.max_discount_amount.max},`
									}



									base.range.usage_limit = {}

									if(base.range.usage_limit.min){
										range.usage_limit.min = `min:${base.range.usage_limit.min},`
									}
									
									if(base.range.usage_limit.max){
										range.usage_limit.max = `max:${base.range.usage_limit.max},`
									}



									base.range.usage_per = {}

									if(base.range.usage_per.min){
										range.usage_per.min = `min:${base.range.usage_per.min},`
									}
									
									if(base.range.usage_per.max){
										range.usage_per.max = `max:${base.range.usage_per.max},`
									}



									base.range.started_at = {}

									if(base.range.started_at.min){
										range.started_at.min = `min:${base.range.started_at.min},`
									}
									
									if(base.range.started_at.max){
										range.started_at.max = `max:${base.range.started_at.max},`
									}



									base.range.expired_at = {}

									if(base.range.expired_at.min){
										range.expired_at.min = `min:${base.range.expired_at.min},`
									}
									
									if(base.range.expired_at.max){
										range.expired_at.max = `max:${base.range.expired_at.max},`
									}
								}
							}

							if(models['deepinfra']){
								var { sql } = await Deepinfra(env.deepinfra, 'openai/gpt-oss-20b', text2json(language, prompt, range, now).trim(), content)

								models['deepinfra'] -= 1

							}else if(gemini_llm_api){
								var { sql } = await Gemini(gemini_llm_api, gemini_llm_model, text2json(language, prompt, range, now).trim(), content)

								models[gemini_llm_api+'-'+gemini_llm_model] -= 1

							}else{
								clear_condition += ` AND "id" != "${task.id}"`

								continue
							}

							if(!sql){
								continue
							}

							if(!sql.where){
								continue
							}


							var generation = ''

							var augmented = ''

							// var ensemble

							// 유료 회원이면 이전 컨텍스트 합쳐서 답변하기
							if(task.topK > 10){
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
												var obj = Object.assign({}, results[r])

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

							if(sql.where.length){
								for(var p = 0; p < sql.where.length; p++){
									var context = sql.where[p]

									context.id = hashId()

									if(!context.type){
										continue
									}


									context.orderBy = "DESC"

									if(context.find){
										// 최근 많이 판매된 가격이 5만 원 이상인 상품만 보여줘

										if(find == 'few' || find == 'little'){
											context.orderBy = "ASC"
										}
									}

									if(context.status){

									}

									/*
										context.type
										context.find
										context.status

									*/


									/*
										가격 필터 UI로 만들어 놓기
										task.condition
											amount
												eq:가격 값
												gte:가격 이상 값
												lte:가격 이하 값
									*/  

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





									if(models['cloudflare']){
										var { data: queryVector } = await env.AI.run('@cf/baai/bge-m3', {
											text: [context.text],
										})

										models['cloudflare'] -= 1

									}else if(models['deepinfra']){
										var queryVector = await Deepinfra(env.deepinfra, 'BAAI/bge-m3', '', context.text)

										var $VectorizeVector: VectorizeVector[] = embeddings.map((values, i) => {
											return {
												id: item.id,
												values: values,
												metadata: metadata
											}
										})

										models['deepinfra'] -= 1

									}else{
										clear_condition += ` AND "id" != "${task.id}"`

										continue
									}



									var condition = `"created_at" < ${now}`

									if(Object.keys(context.condition).length){
										for (const key in context.condition) {
											var value = context.condition[key]

											if (context.condition.hasOwnProperty(key)) {
												if(isNaN(value)){
													query.options.filter[key] = value
												}
												
												// if(key == "amount"){
												// 	if(value.currency){
												// 		task.currency = query.options.filter.currency = value.currency
												// 	}
												// }

												condition += parseCondition(value, key, " AND ")
											}
										}
									}

									var { matches } = await env[`${vectorRegion}-${context.type}`].query(queryVector[0], query.options)

									task.no = page_count.toString() // 채팅 나열 순서

									delete query.options.filter.to

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

											delete matches[m].from
											delete matches[m].to
											delete matches[m].cc
											delete matches[m].bcc
											delete matches[m].ref

											if(matches_condition.length){
												matches_condition += ' OR '
											}

											matches_condition += `("id" = "${match.id}" AND "to" = "${team.id}" AND "created_at" < ${now})`
										}
									}

									var { results } = await env[`${zoneRegion}_${context.type}`].prepare(`SELECT * FROM ${context.type} WHERE ${matches_condition} LIMIT 100`).all()

									if(results.length){
										for(var r = 0; r < results.length; r++){
											var item = results[r]

											var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(item.data))

											var data = JSON.parse(decompressedJsonString)

											if(data){
												if(Object.keys(data).length){
													for (const name in data) {
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

									var { results } = await env[`${zoneRegion}_${context.type}`].prepare(`SELECT * FROM ${type} WHERE ${condition} AND "to" = "${team.id}" AND "created_at" < ${now} LIMIT 300`).all()

									if(results.length){
										for(var r = 0; r < results.length; r++){
											var item = results[r]

											var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(item.data))

											var data = JSON.parse(decompressedJsonString)

											if(data){
												if(Object.keys(data).length){
													for (const name in data) {
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

									if(models['deepinfra']){
										var text = await Deepinfra(env.deepinfra, 'openai/gpt-oss-20b', system, content)

										models['deepinfra'] -= 1

									}else if(gemini_llm_api){
										var text = await Gemini(gemini_llm_api, gemini_llm_model, system, content, {"temperature": 1})

										models[gemini_llm_api+'-'+gemini_llm_model] -= 1

									}else{
										clear_condition += ` AND "id" != "${task.id}"`

										continue
									}

									var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
										text : text,
										json : json
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


					statements[region].push(
						env[region].prepare(`
							DELETE FROM tasks WHERE id = "${task.id}"
						`)
					)

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


					team.data.page_count = page_count

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

					headers.set('Content-Type', 'application/json')

					return new Response(JSON.stringify({
						models : models,
						limits : limits,
						counts : pageCount
					}), {
						headers: { "Content-Type": "text/html; charset=utf-8" },
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