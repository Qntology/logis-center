(async function(){
	var timezoneOffset = new Date().getTimezoneOffset() * 60 * 1000

	const isInIframe = window.self !== window.top;

	if(isInIframe){
		return
	}

		

	try{
		var Chrome = window.location.host.indexOf('commerce.logis.center') == -1

		var app = {
			host : 'commerce.logis.center',
			chrome : Chrome,
			stream : null,
			upload : {},
			storage : {
				set : function(items) {
					if(Chrome){
						return new Promise((resolve, reject) => {
							chrome.storage.local.set(items, () => {
								if (chrome.runtime.lastError) {
									return reject(chrome.runtime.lastError);
								}

								resolve()
							})
						})
					}else{
						for (const key in items) {
							if (items.hasOwnProperty(key)) {
								var value = items[key]

								sessionStorage[key] = JSON.stringify(value)
							}
						}
					}
				},
				get : function(key) {
					if(Chrome){
						return new Promise((resolve, reject) => {
							chrome.storage.local.get(key, (result) => {
								resolve(result)
							})
						})
					}else{
						var cookies = sessionStorage[key]
						return {
							cookies : cookies ? JSON.parse(cookies) : {}
						}
					}
						
				},
				clear : function(){
					if(Chrome){
						return new Promise((resolve, reject) => {
							chrome.storage.local.clear(() => {
								resolve()
							})
						})
					}else{
						sessionStorage.clear()
					}
				}
			},
			filters : {
				page : {},
				origin : window.location.origin,
				interval : ''
			},
			draft : {},
			items : {},
			users : {},
			pages : {},
			block : {
				fetch : false,
				talks : false
			},
			fetch : async function({ url, method = "GET", headers = {}, body = null }) {
				
				var { results, session } = await app.request({ url, method, headers, body })

				session = session ? session : {}

				await app.storage.set({'cookies' : session})

				return { results, session }
			},
			request : async function({ url, method = "GET", headers = {}, body = null }) {
				if(Chrome){
					return new Promise((resolve, reject) => {
						chrome.runtime.sendMessage({ url, method, headers, body }, (response) => {
							const err = chrome.runtime.lastError;

							if (err) {
								reject(err);
							} else {
								resolve(response.json);
							}
						});
					});
					
				}else{
					var option = {
						method: method,
						headers: headers
					}

					if(headers['Content-Encoding'] == 'gzip'){
						if(body){
							if(Object.keys(body).length){
								var arr = pako.gzip(new TextEncoder('utf-8').encode(body), { to: 'arraybuffer' })

								option.body = arr.buffer
							}
						}
					}

					var response = await fetch(url, option)

					var json = await response.json()

					if(json.results){
						if(json.results.length){
							for(var i = 0; i < json.results.length; i++){
								var item = json.results[i]

								if(item.data){
									try{
										var decompressedJsonString = new TextDecoder('utf-8').decode(pako.ungzip(item.data))

										var data = JSON.parse(decompressedJsonString)
									}catch(err){
										var arr = new Uint8Array(item.data)

										var decompressedJsonString = new TextDecoder('utf-8').decode(pako.ungzip(arr.buffer))

										var data = JSON.parse(decompressedJsonString)

									}

									json.results[i].data = data
								}
							}
						}
					}

					console.log('json',json)

					return json
				}
					
			}
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
				if (typeof obj1[key] === 'object' && typeof obj2[key] === 'object') {
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

		function isAlmostEqual(obj1, obj2) {
			// 객체가 비어있는지 체크
			if (!obj1 || !obj2) return false;
			if (Object.keys(obj1).length === 0 || Object.keys(obj2).length === 0) return false;

			const keys1 = Object.keys(obj1);
			const keys2 = Object.keys(obj2);

			// 키 개수가 다르면 false
			if (keys1.length !== keys2.length) return false;

			let diffCount = 0;

			for (const key of keys1) {
				if (obj2.hasOwnProperty(key)) {
					if (obj1[key] !== obj2[key]) diffCount++;
					if (diffCount > 1) return false; // 1개 이상 다르면 false
				}
			}

			return true; // 다르면 최대 1개까지 허용
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

		function bufferToBase64(buffer) {
			return btoa(String.fromCharCode(...new Uint8Array(buffer)));
		}

		function getZeroUTC(date, day) {
			var date = new Date(date)

			date.setDate(date.getDate() - day)

			date.setUTCHours(0)
			date.setUTCMinutes(0)
			date.setUTCSeconds(0)
			date.setUTCMilliseconds(0)

			return date.getTime() // 'YYYY-MM-DDTHH:mm:ss.sssZ'
		}


		function time2text(date) {
			date = new Date(date);

			var seconds = Math.floor((new Date() - date) / 1000);

			var interval = seconds / 31536000;

			if (interval > 1) {
				return Math.floor(interval) + " years";
			}
			interval = seconds / 2592000;
			if (interval > 1) {
				return Math.floor(interval) + " months";
			}
			interval = seconds / 86400;
			if (interval > 1) {
				return Math.floor(interval) + " days";
			}
			interval = seconds / 3600;
			if (interval > 1) {
				return Math.floor(interval) + " hours";
			}
			interval = seconds / 60;
			if (interval > 1) {
				return Math.floor(interval) + " minutes";
			}
			return Math.floor(seconds) + " seconds";
		}

		const parseStatus = function(status){
			var step = ''

			if(status == 1){
				step = 'progress'
			}else if(status == 2){
				step = 'stop'
			}else if(status == 3){
				step = 'cancel'
			}else if(status == 4){
				step = 'refund'
			}else if(status == 5){
				step = 'return'
			}else if(status == 6){
				step = 'error'
			}else if(status == 7){
				step = 'expire'
			}else if(status == 8){
				step = 'exchange'
			}else if(status == 9){
				step = 'complete'
			}else if(status == 10){
				step = 'draft'
			}else if(status == 11){
				step = 'show'
			}else if(status == 12){
				step = 'hide'
			}

			return step
		}

		function createSearchParams(params) {
			var searchParams = new URLSearchParams()
			try{
				Object.entries(params).forEach(([key, values]) => {
					if (Array.isArray(values)) {
						values.forEach((value) => {
							searchParams.append(key, value.toString().replace(/ /gi, "%20"))
						})
					} else {
						searchParams.append(key, values.toString().replace(/ /gi, "%20"))
					}
				})
			}catch(err){
				searchParams = err
			}
				
			return searchParams
		}


		const reqUrl = function(cookies, filters, query){
			var params = ``

			if(document.referrer){
				try{
					var referrer = new URL(document.referrer)

					if(window.location.href.indexOf(referrer) > -1){
						params = `&referrer=${encodeURIComponent(document.referrer)}`
					}
				}catch(err){

				}
			}


			if(!query){
				query = {}
			}



			if(!query.type){
				if(filters.type){
					params += `&type=${filters.type}`
				}
			}

			

			var created_at = new Date(new Date().getTime() - timezoneOffset).getTime()

			if(filters.created_at){
				created_at = filters.created_at
			}

			var _href = window.location.href

			if(Object.keys(filters).length){
				if(Object.keys(filters.page).length){
					if(filters.page.data.link){
						_href = filters.page.data.origin+filters.page.data.link
					}
					

					if(filters.page.id){
						query.id = filters.page.id
					}
				}
			}

			
			var footprint

			try{
				footprint = new URL(_href)

				var origin = footprint.origin;

				var pathname = footprint.pathname;

				var search = footprint.search;

				window.footprint = { 'href' : footprint.href }
			}catch(err){

			}

				
			console.log('query',query);

			if(Object.keys(query).length && cookies){
				if(typeof query.to == "undefined"){
					query.to = hashId(cookies.cc+pathname)
				}

				if(footprint){
					query.href = footprint.href
				}

				

				params += `&${new createSearchParams(query).toString()}`
			}

			console.log('origin',origin)


			return (`https://${app.host}?origin=${encodeURIComponent(origin)}&created_at=${created_at}${params}`).toLowerCase()
		}

		function cleanStates(container = document.body) {
			container.querySelectorAll('input').forEach(input => {
				if (input.type === 'checkbox' || input.type === 'radio') {
					input.removeAttribute('checked');
				}
			});
		}

		function setStates(container = document.body) {
			// 1. input 상태 반영
			container.querySelectorAll('input').forEach(input => {
				if (input.type === 'checkbox' || input.type === 'radio') {
					if (input.checked) {
						input.setAttribute('checked', 'true');
					} else {
						input.setAttribute('checked', 'false');
					}
				} else {
					input.setAttribute('value', input.value); // text, number 등
				}
			});

			container.querySelectorAll('a').forEach(a => {
				if(!a.closest(`[class*="${selector.app}"]`)){
					a.setAttribute('href', a.href)
				}
			});

			// 2. textarea 상태 반영
			container.querySelectorAll('textarea').forEach(textarea => {
				textarea.textContent = textarea.value; // 실제 입력 값 반영
			});

			// 3. select/option 상태 반영
			container.querySelectorAll('select').forEach(select => {
				Array.from(select.options).forEach(option => {
					if (option.selected) {
						option.setAttribute('selected', '');
					} else {
						option.removeAttribute('selected');
					}
				});
			});

			// 4. innerHTML 반환
			return container.innerHTML;
		}


		const Sleep = function(ms) {
			return new Promise(resolve => setTimeout(resolve, ms))
		}

		const clients = [
			"*.cafe24.com",
			"*.makeshop.co.kr",
			"admin.godo.co.kr",
			"*.godo.co.kr",
			"*.firstmall.kr",
			"admin.sixshop.com",
			"sixshop.com",
			"admin.imweb.me",
			"www.imweb.me",
			"*.myshopify.com",
			"sell.smartstore.naver.com",
			"wing.coupang.com",
			"soffice.11st.co.kr",
			"scm.gmarket.co.kr",
			"scm.auction.co.kr",
			"seller.interpark.com",
			"seller.wemakeprice.com",
			"sell.ssg.com",
			"marketplus.co.kr",
			"admin.shopby.co.kr",
			"creators.kakaomakers.com",
			"sell.storefarm.naver.com",
			"partner.wemakeprice.com",
			"activeitzone.com",
			"demofran.com"
		]

		const admins = [
			"*.cafe24.com",              // 카페24: myshop.cafe24.com
			"*.makeshop.co.kr",          // 메이크샵: aaa.makeshop.co.kr
			"*.godomall.com",            // 고도몰: legacy 호스트
			"*.godo.co.kr",              // 고도몰2 기반 일부
			"*.firstmall.kr",			 // 퍼스트몰 
			"*.sixshop.com",             // 식스샵: shop.sixshop.com, abc.sixshop.com
			"*.imweb.me",                // 아임웹: yourbrand.imweb.me
			"*.myshopify.com",           // Shopify: global
			"*.shopby.co.kr",            // 샵바이: NHN 커머스
			"*.wisa.co.kr",              // 위사: 일부 고도몰 파트너사
			"*.sellstore.co.kr",         // 일부 커머스 솔루션
			"*.squarespace.com",         // 스퀘어스페이스 (글로벌이나 일부 국내 사용)
			"*.storefarm.naver.com",     // 스토어팜(구 네이버 쇼핑몰)
			"*.smartstore.naver.com",    // 스마트스토어
			"*.gmkt.kr",                 // G마켓 모바일 단축 도메인
			"*.gmarket.co.kr",           // G마켓
			"*.auction.co.kr",           // 옥션
			"*.interpark.com",           // 인터파크
			"*.wemakeprice.com",         // 위메프
			"*.ssg.com",                 // SSG
			"*.coupang.com",             // 쿠팡
			"*.11st.co.kr",              // 11번가
			"*.kakaomakers.com",          // 카카오메이커스
			"*.activeitzone.com",
			"*.demofran.com"
		]


		function isShop(href, urls) {
			const parsed = new URL(href)
			return urls.some(pattern => {
				const regex = new RegExp("^" + pattern.replace(".", "\\.").replace("*", ".*") + "$")
				return regex.exec(parsed.hostname)
			})
		}


		function item2html(item, checked, extend){
			var href = ''

			if(item.data){
				if(item.data.link){
					href = item.data.link
				}	
			}

			var footprint = new URL(window.location.href.toLowerCase())

			footprint.href = footprint.href.toLowerCase()

			var footprint_params = footprint.searchParams;

			var pathname = footprint.pathname
			var search = footprint.search
			var link = pathname + search 

			var more = false

			if(href){
				var item_url = new URL(footprint.origin+href)
				var item_params = item_url.searchParams;

				var item_obj = Object.fromEntries(item_params.entries());

				var footprint_obj = Object.fromEntries(footprint_params.entries());

				if(isAlmostEqual(item_obj, footprint_obj) || footprint.href.indexOf(link)){
					more = true
				}
			}


			var body = `<input type="checkbox" id="more-${item.id}" ${checked ? 'disabled' : ''} ${checked ? 'checked' : ''} /><div class="${selector.result}">`;

			var itemType = item.type

			if(item.type == "sales"){
				itemType = "sales"
				item.type = "order"

			}else if(item.type == "goods" || item.type == "order"){
				itemType = "sales"

			}else if(item.type == "event" || item.type == "coupon"){
				itemType = "event"

			}else if(item.type == "receiving" || item.type == "shipping"){
				itemType = "tracking"

			}

			function Tpl(item, key, unit){
				var _value = ''

				var _unit = ''

				var _name = key.replace(/_/gi, " ")


				if(typeof item[key] != "undefined"){
					_value = item[key]
				}else if(item.data){
					if(typeof item.data[key] != "undefined"){
						_value = item.data[key]
					}
				}


				if(_value && key == "status"){
					_value = parseStatus(_value)
				}

				if(unit){
					if(typeof item[unit] != "undefined"){
						_unit = ` (${item[unit]})`
					}else if(item.data){
						if(typeof item.data[unit] != "undefined"){
							_unit = ` (${item.data[unit]})`
						}
					}
				}

				var props = ''

				var tagName = 'div'

				if(key == 'title'){
					tagName = 'a'

					if(item.data){
						if(item.data.link){
							props = `href="${item.data.origin}${item.data.link}" target="_blank"`
						}
					}
				}

				if(key == "created_at" || key == "updated_at" || key == "started_at" || key == "expired_at"){
					_value = time2text(_value)

					if(key == "created_at"){
						_name = _value
						_value = `<label for="more-${item.id}">more</label>`
					}
				}

				if(key == "status"){
					_name = item.type
				}

				if(key != "created_at"){
					var input_type = 'text'

					// console.log('typeof _value == "string"',typeof _value == "string");
					// console.log('key,_value',key,_value);

					if(typeof _value == "string"){
						_value = _value.replace(/\\/g, '\\\\');

						// 1. & (앰퍼샌드)를 먼저 처리해야 &quot; 등이 &&quot;로 잘못 변환되는 것을 방지합니다.
						_value = _value.replace(/&/g, '&amp;');

						// 2. < (보다 작음) - 태그 시작 방지 (XSS 방어)
						_value = _value.replace(/</g, '&lt;');

						// 3. > (보다 큼) - 태그 끝 방지 (XSS 방어)
						_value = _value.replace(/>/g, '&gt;');

						// 4. " (큰따옴표) - 속성 값 충돌 방지
						_value = _value.replace(/"/g, '&quot;');

						// 5. ' (작은따옴표) - 속성 값이 작은따옴표로 감싸져 있을 때 충돌 방지
						_value = _value.replace(/'/g, '&#39;');

						if(key.indexOf('date') > -1){
							input_type = 'date'
						}
					}else{
						input_type = 'number'
					}

					_value = `<input type="${input_type}" value="${_value}">`
				}
					


				return `<${tagName} ${props} class="${selector.info} ${key}">
					<strong>${_name}</strong>
					<span>${_value}<i>${_unit}</i></span>
				</${tagName}>`
			}


			if(itemType == "sales"){
				body += `
					${Tpl(item,"status")}
					${Tpl(item,"title")}
					${Tpl(item,"sale_price","currency")}
					${Tpl(item,"created_at")}

				<div class="more-${item.id}">
				`
				if(more){
					body += `
						${Tpl(item,"price","currency")}
						${Tpl(item,"quantity")}
						${Tpl(item,"width")}
						${Tpl(item,"height")}
						${Tpl(item,"length")}
						${Tpl(item,"weight")}
						${Tpl(item,"supply_price","currency")}
						${Tpl(item,"discount","currency")}
						${Tpl(item,"reward_point")}
						${Tpl(item,"shipping_fee","currency")}
						${Tpl(item,"shipping_method")}
						${Tpl(item,"shipping_duration")}
						${Tpl(item,"tax_included")}
						${Tpl(item,"release_date")}
						${Tpl(item,"manufacture_date")}
						${Tpl(item,"expiration_date")}
					`
				}

				body += `</div>`

			}else if(itemType == "tracking"){
				item.status = parseStatus(item.status)

				body += `
					${Tpl(item,"status")}
					${Tpl(item,"text")}
					${Tpl(item,"title")}
					${Tpl(item,"created_at")}
				`

				if(item.data){
					body += `
						${Tpl(item,"sender_name")}
						${Tpl(item,"sender_address")}
						${Tpl(item,"sender_phone")}
						${Tpl(item,"recipient_name")}
						${Tpl(item,"recipient_address")}
						${Tpl(item,"recipient_phone")}
					`
				}else{
					// draft 상태로 표기 해야함
				}



			}else if(itemType == "event"){
				item.status = parseStatus(item.status)

				body += `
					${Tpl(item,"status")}
					${Tpl(item,"title")}
					${Tpl(item,"discount")}
					${Tpl(item,"created_at")}

				<label for="more-${item.id}">more</label>
				<div class="more-${item.id}">
				`
				
				if(more){
					body += `
						${Tpl(item,"code")}
						${Tpl(item,"quantity")}
						${Tpl(item,"usage_per")}
						${Tpl(item,"usage_limit")}
						${Tpl(item,"new_customer_only")}
						${Tpl(item,"min_order_amount")}
						${Tpl(item,"max_discount_amount")}
						${Tpl(item,"first_purchase_only")}
						${Tpl(item,"region_restrictions")}
					`
				}

				body += `</div>`
			}

			body += `<input type="hidden" readonly name="${selector.created_at}" value="${item.created_at}" />`


			body += `<div index="${item.index}" event="${item.event}" views="${item.views}" goods="${item.goods}" tracking="${item.tracking}" class="${selector.relate}"></div>`

			body += `</div>`

			return body;
		}

		const db = new Dexie("logis-center");

		db.version(2).stores({
			items : `
				id,
				type,
				from,
				to,
				cc,
				bcc,
				ref,
				created_at,
				updated_at
			`,
			pages : `
				id,
				type,
				from,
				to,
				cc,
				bcc,
				ref,
				data,
				created_at,
				updated_at
			`,
			crons : `
				id,
				cc,
				bcc,
				job,
				ref,
				created_at,
				updated_at
			`
		});

		marked.setOptions({
			breaks: true
		});


		var sendMessage = async function(req){
			var results = {};
			
			try {
				if(req.select){
					var collection = db[req.select]

					if(req.key){
						collection = collection.where(req.key)
					}

					if(req.value){
						collection = collection.equals(req.value)
					}

					if(req.above){ 
						//  above() (보다 큼: > ) 
						collection = collection.above(req.above)
					}

					if(req.below){
						//  below() (보다 작음: < )
						collection = collection.below(req.below)
					}

					if(req.limit){
						//  below() (보다 작음: < )
						collection = collection.limit(req.limit)
					}

					if(req.orderBy){
						collection = collection.orderBy(req.orderBy)
					}

					if(req.desc){
						collection = collection.reverse()
					}

					results = await collection.toArray();
					

				}else if(req.upsert){
					results = await db[req.upsert].put(req.value);

				}else if(req.delete){
					results = await db[req.delete].put(req.key);
					
				}else if(req.clear){
					results = await db[req.clear].clear();

				}

				return { results : results }

			}catch(error) {
				return { results : results }
			}
		}

		var Upsert = {}

		var Select = {}

		var Delete = {}

		var Clear = {}

		Upsert["items"] = async function(value) {
			var query = {
				upsert : "items",
				value : value
			}

			if(app.chrome){
				return new Promise((resolve, reject) => {
					chrome.runtime.sendMessage(query, (response) => {
						const err = chrome.runtime.lastError;
						if (err) {
							reject(err);
						} else if (response?.results) {
							resolve(response.results);
						} else {
							reject(response?.error || "Unknown error");
						}
					});
				});
			}else{
				var response = await sendMessage(query)

				return response.results
			}

				
		}


		Select["items"] = async function(query) {
			if(typeof query == "undefined"){
				query = {}
			}

			query.select = "items"

			if(app.chrome){
				return new Promise((resolve, reject) => {
					chrome.runtime.sendMessage(query, (response) => {
						const err = chrome.runtime.lastError;
						if (err) {
							reject(err);
						} else if (response?.results) {
							resolve(response.results);
						} else {
							reject(response?.error || "Unknown error");
						}
					});
				});
			}else{
				var response = await sendMessage(query)

				return response.results
			}
				
		}


		Delete["items"] = async function(query) {
			query.delete = "items"

			if(app.chrome){
				return new Promise((resolve, reject) => {
					chrome.runtime.sendMessage(query, (response) => {
						const err = chrome.runtime.lastError;
						if (err) {
							reject(err);
						} else if (response?.results) {
							resolve(response.results);
						} else {
							reject(response?.error || "Unknown error");
						}
					});
				});
			}else{
				var response = await sendMessage(query)

				return response.results
			}
		}


		Clear["items"] = async function(query) {
			if(typeof query == "undefined"){
				query = {}
			}

			query.clear = "items"

			if(app.chrome){
				return new Promise((resolve, reject) => {
					chrome.runtime.sendMessage(query, (response) => {
						const err = chrome.runtime.lastError;
						if (err) {
							reject(err);
						} else if (response?.results) {
							resolve(response.results);
						} else {
							reject(response?.error || "Unknown error");
						}
					});
				});
			}else{
				var response = await sendMessage(query)

				return response.results
			}
		}

		Upsert["pages"] = async function(value) {
			var query = {
				upsert : "pages",
				value : value
			}

			if(app.chrome){
				return new Promise((resolve, reject) => {
					chrome.runtime.sendMessage(query, (response) => {
						const err = chrome.runtime.lastError;
						if (err) {
							reject(err);
						} else if (response?.results) {
							resolve(response.results);
						} else {
							reject(response?.error || "Unknown error");
						}
					});
				});
			}else{
				var response = await sendMessage(query)

				return response.results
			}
		}


		Select["pages"] = async function(query) {
			if(typeof query == "undefined"){
				query = {}
			}

			query.select = "pages"

			if(app.chrome){
				return new Promise((resolve, reject) => {
					chrome.runtime.sendMessage(query, (response) => {
						const err = chrome.runtime.lastError;
						if (err) {
							reject(err);
						} else if (response?.results) {
							resolve(response.results);
						} else {
							reject(response?.error || "Unknown error");
						}
					});
				});
			}else{
				var response = await sendMessage(query)

				return response.results
			}				
		}


		Delete["pages"] = async function(query) {
			query.delete = "pages"

			if(app.chrome){
				return new Promise((resolve, reject) => {
					chrome.runtime.sendMessage(query, (response) => {
						const err = chrome.runtime.lastError;
						if (err) {
							reject(err);
						} else if (response?.results) {
							resolve(response.results);
						} else {
							reject(response?.error || "Unknown error");
						}
					});
				});
			}else{
				var response = await sendMessage(query)

				return response.results
			}
		}


		Clear["pages"] = async function(query) {
			if(typeof query == "undefined"){
				query = {}
			}

			query.clear = "pages"

			if(app.chrome){
				return new Promise((resolve, reject) => {
					chrome.runtime.sendMessage(query, (response) => {
						const err = chrome.runtime.lastError;
						if (err) {
							reject(err);
						} else if (response?.results) {
							resolve(response.results);
						} else {
							reject(response?.error || "Unknown error");
						}
					});
				});
			}else{
				var response = await sendMessage(query)

				return response.results
			}
		}

		Upsert["users"] = async function(value) {
			var query = {
				upsert : "users",
				value : value
			}

			if(app.chrome){
				return new Promise((resolve, reject) => {
					chrome.runtime.sendMessage(query, (response) => {
						const err = chrome.runtime.lastError;
						if (err) {
							reject(err);
						} else if (response?.results) {
							resolve(response.results);
						} else {
							reject(response?.error || "Unknown error");
						}
					});
				});
			}else{
				var response = await sendMessage(query)

				return response.results
			}
		}

		Select["users"] = async function(query) {
			if(typeof query == "undefined"){
				query = {}
			}

			query.select = "users"

			if(app.chrome){
				return new Promise((resolve, reject) => {
					chrome.runtime.sendMessage(query, (response) => {
						const err = chrome.runtime.lastError;
						if (err) {
							reject(err);
						} else if (response?.results) {
							resolve(response.results);
						} else {
							reject(response?.error || "Unknown error");
						}
					});
				});
			}else{
				var response = await sendMessage(query)

				return response.results
			}				
		}


		Delete["users"] = async function(query) {
			query.delete = "users"

			if(app.chrome){
				return new Promise((resolve, reject) => {
					chrome.runtime.sendMessage(query, (response) => {
						const err = chrome.runtime.lastError;
						if (err) {
							reject(err);
						} else if (response?.results) {
							resolve(response.results);
						} else {
							reject(response?.error || "Unknown error");
						}
					});
				});
			}else{
				var response = await sendMessage(query)

				return response.results
			}
		}


		Clear["users"] = async function(query) {
			if(typeof query == "undefined"){
				query = {}
			}

			query.clear = "users"

			if(app.chrome){
				return new Promise((resolve, reject) => {
					chrome.runtime.sendMessage(query, (response) => {
						const err = chrome.runtime.lastError;
						if (err) {
							reject(err);
						} else if (response?.results) {
							resolve(response.results);
						} else {
							reject(response?.error || "Unknown error");
						}
					});
				});
			}else{
				var response = await sendMessage(query)

				return response.results
			}
		}

		Upsert["crons"] = async function(value) {
			var query = {
				upsert : "crons",
				value : value
			}

			if(app.chrome){
				return new Promise((resolve, reject) => {
					chrome.runtime.sendMessage(query, (response) => {
						const err = chrome.runtime.lastError;
						if (err) {
							reject(err);
						} else if (response?.results) {
							resolve(response.results);
						} else {
							reject(response?.error || "Unknown error");
						}
					});
				});
			}else{
				var response = await sendMessage(query)

				return response.results
			}
		}


		Select["crons"] = async function(query) {
			if(typeof query == "undefined"){
				query = {}
			}

			query.select = "crons"

			if(app.chrome){
				return new Promise((resolve, reject) => {
					chrome.runtime.sendMessage(query, (response) => {
						const err = chrome.runtime.lastError;
						if (err) {
							reject(err);
						} else if (response?.results) {
							resolve(response.results);
						} else {
							reject(response?.error || "Unknown error");
						}
					});
				});
			}else{
				var response = await sendMessage(query)

				return response.results
			}
		}


		Delete["crons"] = async function(query) {
			query.delete = "crons"

			if(app.chrome){
				return new Promise((resolve, reject) => {
					chrome.runtime.sendMessage(query, (response) => {
						const err = chrome.runtime.lastError;
						if (err) {
							reject(err);
						} else if (response?.results) {
							resolve(response.results);
						} else {
							reject(response?.error || "Unknown error");
						}
					});
				});
			}else{
				var response = await sendMessage(query)

				return response.results
			}
		}


		Clear["crons"] = async function(query) {
			if(typeof query == "undefined"){
				query = {}
			}

			query.clear = "crons"

			if(app.chrome){
				return new Promise((resolve, reject) => {
					chrome.runtime.sendMessage(query, (response) => {
						const err = chrome.runtime.lastError;
						if (err) {
							reject(err);
						} else if (response?.results) {
							resolve(response.results);
						} else {
							reject(response?.error || "Unknown error");
						}
					});
				});
			}else{
				var response = await sendMessage(query)

				return response.results
			}
		}


		function mergeNode(obj1, obj2) {
			const isEmpty = (value) => value === null || value === undefined || value === '';

			const merged = { ...obj1 };

			for (const key in obj2) {
				if (obj2.hasOwnProperty(key)) {
					const value2 = obj2[key];

					if (!isEmpty(value2)) {
						merged[key] = value2;
					}
				}
			}

			return merged;
		}


		function hashId(text){
			if(typeof text == "undefined"){
				var account = Ethers.Wallet.createRandom()
				text = account.privateKey
			}

			var hashMessage = Ethers.hashMessage(text)

			return Ethers.computeAddress(hashMessage).toLowerCase()
		}

		function randomHash(msg) {
			if(typeof msg == "undefined"){
				msg = Math.random()+""
			}
			return crc32(msg).toString(32)
		}

		console.log('window.location.href',window.location.href)

		var footprint = new URL(window.location.href.toLowerCase())

		footprint.href = footprint.href.toLowerCase()

		var pathname = footprint.pathname

		var host = footprint.host
		
		var href = footprint.href

		var { cookies } = await app.storage.get('cookies')

		var origin = window.location.origin

		if(!cookies?.hash){
			var params = ``

			if(document.referrer){
				try{
					var referrer = new URL(document.referrer)

					if(window.location.href.indexOf(referrer) > -1){
						params = `&referrer=${encodeURIComponent(document.referrer)}`
					}
				}catch(err){

				}
			}


			var footprint = new URL(window.location.href.toLowerCase())

			if(app.filters.page){
				var _page = app.filters.page

				try{
					footprint = new URL(_page.data.origin+_page.data.link)
				}catch(err){

				}
			}

			var origin = footprint.origin

			var created_at = new Date(new Date().getTime() - timezoneOffset).getTime()


			var { results, session } = await app.fetch({
				url : reqUrl( cookies, app.filters, { href : window.location.href } ),
				method: "GET",
				headers: {
					"Content-Type": "application/json"
				}
			})

			cookies = session

			try{
				
				// await app.storage.set({'cookies' : {}})
			}catch(err){
				console.log('err',err);
			}
		}

		console.log('cookies',cookies)





		var $items = []
		var page

		try{
			var pageId = hashId(cookies.team+cookies.cc+pathname)

			console.log('pageId',pageId);
			var pages = await Select['pages']({
				key : 'id',
				value : pageId
			})

			if(pages.length){
				page = pages[0]

				if(page.data){
					if(page.data.item){
						console.log('page.data.item',page.data.item);
						$items = document.querySelectorAll(page.data.item)

						if($items.length == 0){
							var _pages = await Select['pages']({
								key : 'id',
								value : hashId(cookies.cc+pathname.toUpperCase())
							})

							if(_pages.length){
								page = _pages[0]
							}else{
								page = undefined
							}
						}
					}
				}
			}

			var detailId = hashId(cookies.team+cookies.cc+footprint.pathname+footprint.search)

			var details = await Select['items']({
				key : 'ref',
				value : detailId
			})

			console.log('detailId',detailId)

			if(details.length || pages.length == 0){
				console.log('details',details);

				var detail = details[0]

				console.log('detail',detail);

				if(detail.data){
					if(!detail.data.item || detail.data.detail){
						console.log('detail.data.item',detail.data.item);
						var _details = await Select['pages']({
							key : 'id',
							value : detailId
						})


						var $detail = document.querySelectorAll(detail.data.node)

						if(_details.length && $detail.length && $items.length == 0){
							page = _details[0]
						}else{
							page = undefined
						}
					}
				}
			}


			if(page){
				app.filters.page = page
			}

				





			console.log('page',page)
		}catch(err){
			console.log('err',err);
		}
		


		var isMobile = ""

		if (navigator.userAgent) {
			if((/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent))){
				if((/iphone|ipad|ipod/i.test(navigator.userAgent.toLowerCase()))){
					isMobile = "ios"
				}else{
					isMobile = "aos"
				}
			}
		}


		var flag = Intl.DateTimeFormat().resolvedOptions().locale

		var isClient = isShop(href, clients)

		var isAdmin = isShop(href, admins)



		var current = new Date(new Date().getTime() - timezoneOffset).getTime()

		var $current = new Date(current).toISOString()

		var landing = {
			home : randomHash(),
			area : randomHash(),
			headline : randomHash(),
			article : randomHash(),
			section : randomHash(),
			title : randomHash(),
			desc : randomHash(),
			link : randomHash()
		}

		landing.page = `
			<div class="${landing.home}">
				<div id="${landing.headline}">
					<div class="${landing.area}">
						<h2>
							<div style="font-size:1em; font-weight: 100;"><strong>셀러의 시간 절약을 위한</strong><br><strong>편리한 AI 물류관리 시작</strong></div>
						</h2>
						<span><a>AI로 업무 효율은 높이고, 부담은 줄이세요</a></span>
						<a class="${landing.link}">Add To Chrome</a>
					</div>
				</div>
				<div class="${landing.article}">
					<div class="${landing.section}">
						<div class="${landing.title}">
							<span>셀러에게 딱 맞는 시작<br>5분이면 충분합니다</span>
						</div>
						<div class="${landing.desc}">
							<span style="font-weight: 300;">주문, 재고 통합 관리<br>AI와 함께하세요</span>
						</div>
					</div>
				</div>
			</div>
		`

		var selector = {
			mobile : randomHash(),
			desktop: randomHash(),

			left : randomHash(),
			right : randomHash(),
			center : randomHash(),

			aside : randomHash(),

			block : "_"+randomHash(),

			ocr : randomHash(),
			dom : randomHash(),

			sender : '_'+randomHash(),

			markdown : randomHash(),
			
			file : randomHash(),
			reset : randomHash(),


			created_at : randomHash(),
			started_at : randomHash(),
			expired_at : randomHash(),
			index : randomHash(),
			event : randomHash(),
			views : randomHash(),
			goods : randomHash(),
			status : randomHash(),
			width : randomHash(),
			height : randomHash(),
			length : randomHash(),
			weight : randomHash(),
			size : randomHash(),
			currency : randomHash(),
			supply_price : randomHash(),
			sale_price : randomHash(),
			discount : randomHash(),
			quantity : randomHash(),
			tracking : randomHash(),
			number : randomHash(),
			carrier : randomHash(),
			shipping_fee : randomHash(),
			shipping_method : randomHash(),
			shipping_duration : randomHash(),
			fulfillment_service : randomHash(),
			stock_keeping_unit : randomHash(),
			bundle_shipping : randomHash(),
			shipping_date : randomHash(),
			delivery_date : randomHash(),
			order_date : randomHash(),
			payment_date : randomHash(),
			payment_method : randomHash(),
			payment_origin : randomHash(),
			payment_number : randomHash(),
			used : randomHash(),
			lease : randomHash(),
			rental : randomHash(),
			refurbish : randomHash(),
			tax_included : randomHash(),
			release_date : randomHash(),
			
			phone : randomHash(),
			address : randomHash(),
			code : randomHash(),
			usage_per : randomHash(),
			usage_limit : randomHash(),
			new_customer_only : randomHash(),
			min_order_amount : randomHash(),
			max_discount_amount : randomHash(),
			first_purchase_only : randomHash(),
			region_restrictions : randomHash(),

			no : randomHash(),
			sender_address : randomHash(),
			sender_phone : randomHash(),
			recipient_address : randomHash(),
			recipient_phone : randomHash(),

			paginate : randomHash(),
			bottom : randomHash(),

			menu : randomHash(),

			interval : randomHash(),
			recent : randomHash(),
			week : randomHash(),
			month : randomHash(),

			count : randomHash(),

			sign : {
				in : randomHash(),
				out : randomHash(),
			},

			header : randomHash(),

			leave  : randomHash(),
			invite : randomHash(),

			address : "_"+randomHash(),

			profile : randomHash(),

			name : randomHash(),

			favicon : randomHash(),

			setting : randomHash(),



			row : randomHash(),
			col : randomHash(),

			pip : randomHash(),

			dim : randomHash(),
			scan : randomHash(),
			scanning: randomHash(),

			chat : randomHash(),

			loading : randomHash(),

			app : randomHash(),
			area : randomHash(),

			col : randomHash(),

			filters : randomHash(),
			results : randomHash(),
			result :  randomHash(),


			membership : randomHash(),
			member : randomHash(),

			pages : randomHash(),
			page : "page-"+randomHash(),


			users : randomHash(),
			user : randomHash(),

			team : randomHash(),

			branch : randomHash(),

			extend : randomHash(), 
			parent : randomHash(),
			child : randomHash(),


			relate : randomHash(),

			info : randomHash(),

			host : randomHash(),
			type : randomHash(),

			children : randomHash(),

			actions : randomHash(),
			action : randomHash(),

			toggle : randomHash(),

			qrcode : randomHash(),
			qrauth : randomHash(),

			camera : randomHash(),
			vision : randomHash(),
			canvas : randomHash(),
			photo  : randomHash(),
			video  : randomHash(),
			aiocr  : randomHash(),

			prompt : randomHash(),
			context : randomHash(),
			submit :  randomHash(),

			checkbox : randomHash(),
			label : randomHash(),


			icon : randomHash(),

			content : randomHash(),

			message : randomHash(),
			talk : randomHash(),

			$prompt : randomHash(),

			user : randomHash(),
			system : randomHash(),

			type_reset : randomHash(),
			type_draft : randomHash(),
			type_sales : randomHash(),
			type_goods : randomHash(),
			type_order : randomHash(),
			type_event : randomHash(),
			type_coupon : randomHash(),
			type_tracking : randomHash(),


			scroll : randomHash(),
			talks : randomHash(),
			none : randomHash(), 

			hidden : randomHash(),
			$list : randomHash(),
			$item : randomHash(), // 동기화 전
			active : randomHash(), // 동기화 후
			visited : randomHash(), // 상세 동기화 후
			completed : randomHash()
		}

		var isLock = false

		var isFocus


		window.addEventListener("blur", async function(event) {
			if(typeof isFocus != "undefined"){
				isFocus = false

				try{
					if(window[cookies.hash]){
						timeout.clear()
					}
				}catch(err){

				}
			}
		})

		window.addEventListener("focus", async function(event) {
			if(typeof isFocus != "undefined"){
				isFocus = true

				try{
					var { cookies } = await app.storage.get('cookies')

					if(!window[cookies.hash]){
						window[cookies.hash] = setTimeout(timeout.fn, timeout.ms)
					}
				}catch(err){

				}
			}
		})


		const removeChild = (d) => d && d.parentNode && d.parentNode.removeChild(d)




		var retryCount = 0

		var onAuth = async function(e){
			var $qrauth = $app.querySelector(`[class*="${selector.qrauth}"]`)

			timeout.ms = 1000
			retryCount = 1

			$qrauth.setAttribute("style",`line-height: 8px; font-size: 40px; text-decoration: none;`)

			$qrauth.textContent = "."

			var { cookies } = await app.storage.get('cookies')
			
			window[cookies.hash] = setTimeout(timeout.fn, timeout.ms)
		}

		var hostElement = document.createElement('div')

		document.body.appendChild(hostElement)


		var shadowRoot = hostElement.attachShadow({ mode: 'closed' });

		var $app = document.createElement(app.chrome ? "div" : "logis")

			$app.className = `${selector.app} ${isMobile ? selector.mobile : selector.desktop}`
			$app.setAttribute(selector.page, "")
			$app.setAttribute("_"+selector.type,"")


		if(cookies.cc){
			var scanId = hashId(cookies.cc+window.location.pathname+window.location.search)

			var crons = await Select['crons']({
				key : 'ref',
				value : scanId
			})

			if(crons.length){
				$app.classList.add(selector.scanning)
			}
		}

		console.log('selector.scanning',selector.scanning);


		const placeholder = {
			prompt : function(){
				return cookies.address ? "Enter prompt" : "Please scan the QR code to verify"
			},
			confirm : "문서를 동기하시겠습니까?"
		}



		var ExtendStyle = document.createElement("style")

		var ShadowStyle = document.createElement("style")


		ExtendStyle.innerHTML = `
			[class*="${selector.$item}"]{filter: grayscale(1) opacity(0.5) !important;}

			[class*="${selector.active}"]{filter: grayscale(1) !important;}


			[class*="${selector.visited}"]{filter: none !important;}
			[class*="${selector.visited}"] *{font-weight:900 !important;}
			[class*="${selector.visited}"] img{filter: none !important;}

			[class*="${selector.completed}"]{filter: invert(1) !important;}
			[class*="${selector.completed}"] img{filter: invert(1) !important;}
		`


		if(app.chrome){
			ShadowStyle.innerHTML = `
				[class*="${selector.area}"] *,
				[class*="${selector.app}"]{all: initial; background-color: transparent; line-height:1em; font-size:14px; font-weight:500; font-family: sans-serif; color:#fff; pointer-events: none;}
				[class*="${selector.app}"]{position: fixed; right: 0; top: 0; bottom: 0; width: 100%; pointer-events: none; z-index:100000000000000;}
				
				[class*="${selector.row}"]{display: table; width:100%; height:100%;}
				[class*="${selector.col}"]{display: table-cell; width:auto; height:100%;}

				[for="${selector.pip}"]{position:absolute; top: 0; right: 0; z-index:1;}


				[id="${selector.right}"]:checked+[class*="${selector.area}"]:after{backdrop-filter: blur(3px); background: rgba(0, 0, 0, 0.8); z-index: -1;}
				[id="${selector.right}"]:checked+[class*="${selector.area}"] *{pointer-events: initial;}

				[id="${landing.home}"]{display: block;}

				[id="${selector.sign.in}"]{display: none; font-size:12px;}
				[id="${selector.sign.out}"]{display: inline-block; font-size:12px;}

				[class*="${selector.app}"][${selector.address}=""] [id="${selector.right}"]:checked+[class*="${selector.area}"] [id="${selector.sign.out}"]{display: none;}
				[class*="${selector.app}"][${selector.address}=""] [id="${selector.right}"]:checked+[class*="${selector.area}"] [id="${selector.sign.in}"]{display: inline-block;}

				[class*="${selector.app}"] [id="${selector.right}"]:checked+[class*="${selector.area}"]+[for="${selector.left}"]{display:none;}

				[id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.header}"]{z-index: 9; opacity:1;}
				[id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.center}"]{z-index: 0; opacity:1;}
				[id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.left}"]{z-index: 0; opacity:1;}
				[id="${selector.right}"]:checked+[class*="${selector.area}"] [id="${selector.scan}"]{display:none;}

				[id="${selector.header}"]{display: none;}
				[id="${selector.header}"]+[class*="${selector.header}"]{}
				[class*="${selector.header}"]{position:absolute; top: 0; left: 0; right: 0;  height:40px; z-index:1; background-color: #000; z-index: -1; opacity:0; transition-duration:0.3s; box-shadow: 0 0 10px 0px #000;}

				[class*="${selector.header}"] h1[class*="${selector.menu}"]{position:absolute; left:0.5em; top: 0; text-transform:inherit;}
				[class*="${selector.header}"] a[class*="${selector.menu}"]{position:absolute; right:0.5em; top: 2px;}

				[class*="${selector.center}"]{position:fixed; left:265px; right:400px; top: 40px; bottom:0; z-index: -1; opacity:0;}

				[class*="${selector.left}"]{position:fixed; left:0; top: 40px; bottom: 0; left: 0; max-width:265px; width:265px; z-index: -1; opacity:0; transition-duration:0.3s;}
				[class*="${selector.left}"]::-webkit-scrollbar {height: 3px;}
				[class*="${selector.left}"]::-webkit-scrollbar-thumb {background-color: rgba(255,255,255,0.5);}
				[class*="${selector.left}"]::-webkit-scrollbar-track {}

				[class*="${selector.right}"]{position:fixed; top: 40px; bottom: 0; right: 0; min-width: 400px; max-width: 400px; width: 400px; z-index: 9;}

				[class*="${selector.filters}"]{display:block; position: relative; vertical-align: top; background-color:#111;}
				[class*="${selector.filters}"] [class*="${selector.scroll}"]{display: flex; flex-flow: column; padding: 0 0 150px; height: 100%; box-sizing:border-box;}

				[id="${selector.system}"]{display:none;}
				[id="${selector.system}"]:checked+[class*="${selector.chat}"] [for="${selector.system}"]{opacity:1;}
				[id="${selector.system}"]:checked+[class*="${selector.chat}"] [for="${selector.system}"]:before{filter: none;}

				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.system}"]{display:none;}
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.system}"] deco{opacity:0.5;}
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"]:hover deco{opacity:1;}
				

				[id="${selector.system}"]:checked+[class*="${selector.chat}"] [class*="${selector.talk}"]{display:none;}
				[id="${selector.system}"]:checked+[class*="${selector.chat}"] [class*="${selector.talk}"][class*="${selector.system}"]{display:block;}
				[for="${selector.system}"]{position:absolute; top: -29px; right: 1em;opacity:0.5; cursor:pointer;}
				[for="${selector.system}"]:before{content:"✅"; margin-right: 5px; filter: sepia(100%) saturate(100%) grayscale(1);}
				



				[class*="${selector.pip}"]{display: block;}

				[name="${selector.interval}"]{height: 40px; line-height: 40px; appearance: base-select;}
				[name="${selector.interval}"] option{color:#000;}

				[id="${selector.interval}"]{position: relative; display: none; vertical-align: top;}
				[id="${selector.interval}"]:before{content: "/"; margin-left: 13px; margin-right: 13px; font-size: 12px; opacity:0.5;}

				[id="${selector.page}"]{position: relative; display: none; line-height: 40px; vertical-align: top; font-weight:bold; text-transform: capitalize;}
				[id="${selector.page}"]:before{content: "/"; margin-right: 13px; font-size: 12px; opacity:0.5;}
			
				[id="${selector.page}"]:not(:empty),
				[id="${selector.page}"]:not(:empty)+[id="${selector.interval}"]{display: inline-block;}


				[class*="${selector.profile}"]{position:absolute; bottom: 0.4em; left: 1em; right: 1em; z-index: 1;}
				[class*="${selector.profile}"]:after{content: ""; position: absolute; bottom: -1em; left: 0; right: 0; top: 1em; background-color: #111; z-index: -1;}
				[class*="${selector.profile}"] [class*="${selector.menu}"]{cursor:pointer;}
				
				[class*="${selector.profile}"] [class*="${selector.menu}"][id="${selector.sign.in}"]{}
				[class*="${selector.profile}"] [class*="${selector.menu}"][id="${selector.sign.out}"]{opacity:0.5;}


				[class*="${selector.profile}"] [class*="${selector.info}"]{position:relative; overflow:hidden; display: block; padding:0.7em; border-radius:1em; min-height: 40px; background-color:#000; box-shadow: 0 0 20px 0px #000; z-index:0;}

				[class*="${selector.favicon}"]{display: block; position: absolute; left:0; right:0; top: 0; bottom:0; background-repeat: repeat; background-position: center; z-index:-1; visibility:initial;}
				[class*="${selector.name}"]{position: relative; display: block; height:40px; line-height:40px;}

				[name="${selector.setting}"]{position:absolute; top: 0; left: 0; right: 0; bottom: 0; opacity:0; padding-right:50px; z-index:-1; background-color:#fff;}
				[name="${selector.setting}"] input[type="text"]{display: block; width:100%; height: 40px; box-sizing:border-box; color:#000;}
				[for="${selector.setting}"]{text-decoration: underline;}

				
				[class*="${selector.app}"][${selector.address}=""] [class="${selector.info}"],
				[class*="${selector.app}"][${selector.address}=""] [class="${selector.name}"]{display: none;}


				[class*="${selector.profile}"] [type="submit"]{position: absolute; top: 62px; right: 0; font-size: 12px; font-weight: 900; z-index: -1; color: #ffffff; text-decoration: underline; opacity: 0;}

				[class*="${selector.profile}"] [for="${selector.setting}"]{position:absolute; top: 10px; right: 0px; font-size:14px; z-index:2; background-color:#000;}

				[class*="${selector.name}"] strong{line-height: 39px; font-size:16px; background-color:#000;}

				[id="${selector.setting}"]:checked+[class*="${selector.profile}"] [type="submit"]{z-index:1; opacity:1;}

				[id="${selector.setting}"]:checked+[class*="${selector.profile}"] [class*="${selector.info}"]{overflow:initial; color:#000; background-color:#fff;}

				[id="${selector.setting}"]:checked+[class*="${selector.profile}"] [name="${selector.setting}"]{opacity:1; z-index:5;}

				[id="${selector.setting}"]:checked+[class*="${selector.profile}"] [for="${selector.setting}"]{right: initial; left: -8px; top: -35px; font-size: 0; z-index: 7; background-color: rgba(0, 0, 0, 0);}
				[id="${selector.setting}"]:checked+[class*="${selector.profile}"] [for="${selector.setting}"]:after{content: 'Cancel'; font-size: 12px; font-weight: 700; text-decoration:underline;}


				[id="${selector.setting}"]:checked+[class*="${selector.profile}"] [class*="${selector.favicon}"]{display:none;}


				[id="${selector.setting}"]{display:none;}
				
				[id="${selector.setting}"]:checked+[class*="${selector.profile}"] [class*="${selector.menu}"]{position:relative; z-index:-1; opacity:0; pointer-events:none;}






				[class*="${selector.results}"]{position: relative; background-color: rgba(0, 0, 0, 0.5);}

				[class*="${selector.results}"] [class*="${selector.scroll}"]{width:100%; height:100%; padding: 0 9px 200px 1em; box-sizing:border-box; background:#222; background-image: radial-gradient(circle at center, #ffffff44 1px, transparent 0); background-size: 30px 30px; background-position: 0 0, 30px 30px; background-attachment: local; background-repeat: repeat;}
				[class*="${selector.results}"] [class*="${selector.scroll}"]:empty:after{content:'Empty'; position:absolute; left:0; right:0; top:50%; margin-top:-20px; font-size:2em; text-align:center; color:#fff; opacity:0.5;}
				[class*="${selector.results}"] [class*="${selector.scroll}"]:empty:after{}
				[class*="${selector.app}"][class*="${selector.loading}"] [class*="${selector.results}"] [class*="${selector.scroll}"]:empty:after{content:'Loading'; opacity:1;}
				

				[class*="${selector.results}"] [id*="more-"]{display:none;}
				
				[class*="${selector.results}"] [id*="more-"]:checked+[class*="${selector.result}"] [class*="${selector.info}"]{min-height:32px;}
				[class*="${selector.results}"] [id*="more-"]:checked+[class*="${selector.result}"] input{padding:0.5em 0.5em 0.5em 100px; background: #ddd; color: #000; font-weight:900; text-overflow: ellipsis; pointer-events:initial;}
				[class*="${selector.results}"] [id*="more-"]:checked+[class*="${selector.result}"] strong{padding: 0.9em 0; font-size: 11px; color:#000;}
				[class*="${selector.results}"] [id*="more-"]:checked+[class*="${selector.result}"] span i{top:9px; color:#000;}
				[class*="${selector.results}"] [id*="more-"]:checked+[class*="${selector.result}"] [class*="more-"]{display:block; margin-top:0;}
				[class*="${selector.results}"] [id*="more-"]:checked+[class*="${selector.result}"] [for*="more-"]:before{content:"▲ fold"; display: inline-block; vertical-align: top; font-size:14px; color:#000; text-decoration: underline;}
				[class*="${selector.results}"] [id*="more-"]:checked+[class*="${selector.result}"] [for*="more-"]{line-height:32px; font-size:0; background: #ffff00aa;}
				[class*="${selector.results}"] [id*="more-"][disabled]:checked+[class*="${selector.result}"] [for*="more-"]{background: #ddd;}
				[class*="${selector.results}"] [id*="more-"][disabled]:checked+[class*="${selector.result}"] [for*="more-"]:before{display:none;}
				

				[class*="${selector.result}"]{position: relative; overflow:hidden; display:block; margin: 2em auto 0; max-width: 500px;}

				[class*="${selector.result}"] a{position: relative; font-weight:bold; text-decoration:underline; z-index:2; cursor:pointer;}
				[class*="${selector.result}"] a *{cursor:pointer;}

				[class*="${selector.result}"] [class*="more-"]{margin-top:1em;}
			
				[class*="${selector.result}"] [for*="more-"]{display:block; padding-left:100px; cursor:pointer; text-decoration: underline;}
				[class*="${selector.result}"] [for*="more-"]:after{content:''; display:none; position:absolute; top: 0; left:0; right:0; bottom:0; transform:scale(100); z-index:1;}
				[class*="${selector.result}"] [for*="more-"]:before{content:"▼ "}

				[class*="${selector.results}"] [class*="more-"]{display:none;}

				[class*="${selector.results}"] [class*="${selector.talk}"]{display: block; padding:0.5em 1em; cursor:pointer;}
				[class*="${selector.results}"] [class*="${selector.talk}"] *{cursor:pointer;}
				[class*="${selector.results}"] [class*="${selector.talk}"]+[class*="${selector.talk}"]{margin-bottom:1em;}
				[class*="${selector.results}"] [class*="${selector.talk}"]>div{}
				[class*="${selector.results}"] [class*="${selector.talk}"] div{}

				[class*="${selector.result}"][class*="${selector.extend}"]{margin-top:0;}
				[class*="${selector.result}"][class*="${selector.extend}"] [class*="more-"]{display:block; margin-top:0;}
				[class*="${selector.result}"][class*="${selector.extend}"] [class*="${selector.created_at}"]{display:none;}
				[class*="${selector.result}"][class*="${selector.extend}"] input{padding:0.5em 0.5em 0.5em 100px; background: #ccc; font-weight:900; text-overflow: ellipsis; pointer-events:initial;}
				[class*="${selector.result}"][class*="${selector.extend}"] strong{padding: 0.9em 0; font-size: 11px;}
				[class*="${selector.result}"][class*="${selector.extend}"] span i{top:9px; color:#000;}
				[class*="${selector.result}"][class*="${selector.extend}"] [for*="more-"]{line-height:32px; font-size:0; background:#ccc;}



				[class*="${selector.result}"] [class*="${selector.info}"]{position:relative; display: block; min-height:17px; background:#00000050;}
				[class*="${selector.result}"] [class*="${selector.info}"] strong{position:absolute; top: 1px; left:10px; font-size:12px; letter-spacing: -1px; text-transform: capitalize; opacity:0.5; z-index:1;}
				[class*="${selector.result}"] [class*="${selector.info}"] input{padding-left:100px; width:100%; box-sizing:border-box; pointer-events:none;}
				[class*="${selector.result}"] [class*="${selector.info}"] span{position:relative; display: block;}
				[class*="${selector.result}"] [class*="${selector.info}"] span i{position:absolute; right:1.5em; top:3px; font-style:italic; pointer-events:none;}
				
				[class*="${selector.result}"] .created_at{background: #ffff00aa;}
				[class*="${selector.result}"] .created_at strong,
				[class*="${selector.result}"] .created_at label{color:#000;}

				[class*="${selector.xcroll}"]::-webkit-scrollbar {width: 5px;}
				[class*="${selector.xcroll}"]::-webkit-scrollbar-thumb {background-color: rgba(255,255,255,0.5);}
				[class*="${selector.xcroll}"]::-webkit-scrollbar-track {}

				
				[class*="${selector.host}"]{display: inline-block; vertical-align: top;}

				[name="${selector.host}"]{display: block; padding: 0 1em; width: 100%; height: 40px; line-height: 40px; font-size: 1em; font-weight: 900; box-sizing:border-box; background-color:transparent;}
				[name="${selector.host}"] option{color:#000;}


				[id="${selector.filters}"] [class*="${selector.filters}"]{min-width:265px; height:100%;}


				[class*="${selector.area}"]{position: absolute; right: 0; bottom: 0; max-width: 100%; width: 100%; height: 100%; z-index:0;}
				[class*="${selector.area}"]:after{content: ""; position: absolute; right: 0; bottom: 0; width: 100%; height: 100%; backdrop-filter: blur(0); background: rgba(0, 0, 0, 0); z-index: -1; transition-duration:0.3s;}

				[class*="${selector.qrcode}"]{display: none; position: absolute; left: 0; right: 0; bottom: 70px; margin: 10px auto 3px; padding: 1em; border-radius: 1em; max-width: 300px; box-sizing: border-box; background-color: #fff;}
				[class*="${selector.qrcode}"] canvas{display:none;}
				[class*="${selector.qrcode}"] img{display:block; width:100%; max-width:100%; min-width: 100%;}
				[class*="${selector.qrauth}"]{display: none; position: absolute; left: 0; right: 0; bottom: 17px; max-width: 300px; margin: 0 auto; border-radius: 1em; height: 36px; line-height: 36px; background-color: #fff; color: #000000; font-weight: 900; text-transform: uppercase; text-decoration: underline; text-align: center; cursor: pointer; box-shadow: 0 0 20px 0px #00000055;}


				[class*="${selector.chat}"]{display: none; padding-bottom:105px; width: 100%; height: 100vh; box-sizing: border-box;}
				[class*="${selector.chat}"]:after{content: ""; position: absolute; left: 36px; right: 30px; bottom: 60px; box-shadow: 0px 0 30px 10px #000; z-index: -1;}



				[class*="${selector.prompt}"]{position: absolute; left:0; right:0; bottom: 0; display: block; margin:1em; z-index:10;}


				[class*="${selector.app}"][${selector.address}=""] [id="${selector.interval}"],
				[class*="${selector.app}"][${selector.address}=""] [name="${selector.prompt}"]{display:none;}


				[name="${selector.prompt}"][style]{position:relative; height:320px;}
				[name="${selector.prompt}"][style]:after{content: ""; position: absolute; left: 0; right: 0; bottom: 0; box-shadow: 0 0 50px 50px #00000099; z-index: -1;}
				[name="${selector.prompt}"][style] [for="${selector.file}"],
				[name="${selector.prompt}"][style] [id="${selector.reset}"],
				[name="${selector.prompt}"][style]+[for="${selector.address}"]{color:#fff;}

				[name="${selector.prompt}"][style] [id="${selector.reset}"]{color:#fff;}
				[name="${selector.prompt}"][style] [for="${selector.file}"],
				[name="${selector.prompt}"][style]+[class*="${selector.sender}"],
				[name="${selector.prompt}"][style] textarea{display:none;}
				[name="${selector.prompt}"]{overflow: hidden; position: relative; display: block; margin:1em auto; max-width: 500px; height: 125px; border-radius: 10px; background-color: #fff; background-repeat: no-repeat; background-size: contain; background-position: center; box-shadow: 0 0 20px 0px #00000055; z-index:2;}
				[name="${selector.prompt}"] textarea{color: #000;}

				[name="${selector.context}"]{overflow: hidden; overflow-y: scroll; display: block; margin-bottom:45px; padding: 0.5em 1em 0; width: 100%; height: 80px; line-height: 1.2; white-space: pre-wrap; overflow-wrap: break-word; box-sizing: border-box; color: #000;}

				[name="${selector.context}"]::-webkit-scrollbar {width: 5px;}
				[name="${selector.context}"]::-webkit-scrollbar-thumb {background-color: rgba(255,255,255,0.5);}
				[name="${selector.context}"]::-webkit-scrollbar-track {}

				[for="${selector.submit}"]{overflow: hidden; position: absolute; right: 10px; bottom: 10px; border-radius: 50%; cursor:pointer; z-index: 1;}
				[for="${selector.submit}"] img{display:block; width: 25px; height: 25px; cursor:pointer;}
				[id="${selector.submit}"]{display:none;}

				[class*="${selector.app}"] [id="${selector.submit}"]:after{content:""; position:absolute; left:0; top:0; width:100%; height:100%; border-radius:50%; z-index: 0; background-color:#fff;}
				[class*="${selector.app}"] [id="${selector.submit}"]:before{content:""; position:absolute; left:0; top:0; right:0; bottom:0; margin:auto; width:50%; height:50%; z-index: 1; background-color:#000;}


				[class*="${selector.hidden}"]{position: absolute; right: 0; bottom:0; width:100%; height: 100%; z-index: -1;}

				[for="${selector.right}"]{position: absolute; right: 12px; bottom: 16px; margin: 0 auto; border-radius: 15px; width: 36px; height: 36px; text-align: center; font-size: 25px; font-family: 'Noto Color Emoji'; box-shadow: 0 0 4px #000; background: #000000d6; cursor: pointer; z-index: 100; pointer-events: initial; transform:scale(1.3); transition-duration: 0.2s;}
				[for="${selector.right}"]:after{content:"✨"; display: block; text-indent: 1.0px; line-height: 32px; font-size:15px; transition-duration: 0.2s; background-size:32px; background-position:center;}
				
				[class*="${selector.app}"][${selector.page}] [for="${selector.right}"]:after{filter: none;}
				[class*="${selector.app}"][${selector.page}=""] [for="${selector.right}"]:after{filter: sepia(100%) saturate(100%) grayscale(1); color:#ffff00aa;}
				



				[class*="${selector.pages}"] [class*="${selector.branch}"] input{display:none;}
				[class*="${selector.pages}"] [class*="${selector.children}"]{}

				[class*="${selector.pages}"] strong{display: block; margin-bottom: 1em; font-weight: 300; font-size: 10px; font-style: italic; font-family: arial; vertical-align: top;}
				[class*="${selector.pages}"] [class*="${selector.label}"] strong{display: table-caption; margin: 0.3em 0 0; width: 80px; line-height:1.2; font-size:10px; font-weight:300; font-style:italic; opacity:0.5;}

				[class*="${selector.branch}"] [class*="${selector.active}"]{filter :none;}
				[class*="${selector.branch}"] [class*="${selector.active}"]>span{text-decoration: underline; background-color:#fff;}
				[class*="${selector.branch}"] [class*="${selector.active}"]>span span{color:#000;}
				[class*="${selector.branch}"] [class*="${selector.active}"]>span:before{content: "▶"; position: absolute; left: -2em; top: 1px; color: #fff; font-size: 6px;}
				[class*="${selector.branch}"] [class*="${selector.active}"]>span u{color:#000;}
				
				[class*="${selector.branch}"] [class*="${selector.label}"]{width: 80px;}
				[class*="${selector.branch}"] [class*="${selector.label}"] span>span:first-child{display:none;}
				[class*="${selector.branch}"] [class*="${selector.label}"]:last-child span>span:first-child{display:inline;}
				[class*="${selector.branch}"] [class*="${selector.label}"] + [class*="${selector.child}"] span>span{display:inline;}

				
				[class*="${selector.filters}"] label span{position: relative; display: inline-block; vertical-align:top; text-transform: capitalize; white-space: nowrap; font-weight: 300;}
				[class*="${selector.filters}"] label span:after{content:''; position:absolute;}
				[class*="${selector.filters}"] label:hover>span{background-color:#fff;}
				[class*="${selector.filters}"] label:hover span span{color:#000; text-decoration: underline;}
				[class*="${selector.filters}"] label:hover span u{color:#000;}

				[class*="${selector.pages}"] [class*="${selector.children}"]>label>span{font-family: arial;}

				[class*="${selector.pages}"]{position: relative; display: block; margin: 1em 1em 0;}
				[class*="${selector.pages}"] [class*="${selector.parent}"]{overflow:initial;}
				[class*="${selector.pages}"]>[class*="${selector.branch}"]>li[class*="${selector.parent}"]{border-bottom: 1px dashed #434343; margin-bottom: 1.5em;}


				[class*="${selector.scan}"]{display:none;}

				[id*="page-"]:checked>[page-id] div[id*="record-"]{display:block;}

				div[id*="record-"]{display:none; margin-left:2em;}
			


				[class*="${selector.paginate}"]{margin-top:1em; display:block; text-align:center;}
				

				
				[class*="${selector.membership}"],
				[id="${selector.membership}"]{display:none;}
				[id="${selector.membership}"]:checked +[id="${selector.filters}"] [class*="${selector.membership}"]{display:block;}
				[id="${selector.membership}"]:checked +[id="${selector.filters}"] [for="${selector.membership}"]{font-size:0;}
				[id="${selector.membership}"]:checked +[id="${selector.filters}"] [for="${selector.membership}"]:after{content:'back'; font-size:10px; text-decoration: underline;}
				[for="${selector.membership}"]{position: absolute; left: 60px; top: -1px; font-size: 10px; font-weight: 300; text-decoration: underline;}



				[class*="${selector.users}"]{display: block; margin: 2em 1em 5em;}
				[class*="${selector.users}"] [class*="${selector.branch}"] [class*="${selector.label}"]{min-width:80px; width:auto;}


				[name="${selector.team}"]{display:none;}

				[id*="team-"]:checked+[class*="${selector.parent}"] [class*="team-"] [name="${selector.team}"]{display:block;}

				[team-id]{position: relative; display: block;}
				
				[team-id]>strong{float:left; margin-bottom: 0.5em; font-size:10px; font-weight:300; font-style: italic;}
				[team-id]>label{display:inline-block; font-size:10px; margin-bottom: 0; margin-left: 9px; padding:0; text-decoration:underline; vertical-align: top;}
				

				[user-id] [class*="${selector.label}"] i,
				[team-id] [class*="${selector.label}"] i{margin-left:5px; font-size:9px; letter-spacing:-0.5px; vertical-align: top;}



				[class*="${selector.parent}"]{position: relative; display: block; overflow:hidden;}
				[class*="${selector.child}"]{position: relative; display: inline-block; float: left; clear: both;}


				[id="${selector.invite}"]{display:none;}
				[id="${selector.invite}"]:checked+[class*="${selector.filters}"] [class*="${selector.users}"] label[class*="${selector.label}"][class*="user-"]{display:none;}

				[id="${selector.invite}"]:checked+[class*="${selector.filters}"] [class*="${selector.users}"] [class*="${selector.invite}"]{display:block;}

				[id="${selector.invite}"]:checked+[class*="${selector.filters}"] [class*="${selector.users}"] [for="${selector.invite}"]{position: absolute; bottom: -35px; left: 0; right: 0; font-size: 0; height: 30px;}
				[id="${selector.invite}"]:checked+[class*="${selector.filters}"] [class*="${selector.users}"] [for="${selector.invite}"]:after{content:"cancel"; display:block; font-size:14px; line-height:2; text-align:center;}


				label[for="${selector.invite}"]{margin-top: 1em; text-decoration: underline; font-size: 10px; text-indent: 3px;}

				[class*="${selector.invite}"]{overflow:hidden; position: relative; display: none; margin-top: 1em; border:1px solid rgba(255,255,255,0.1); border-radius:1em;}
				[class*="${selector.invite}"] input[type="email"]{width:100%; height:30px; padding:0 4px; box-sizing:border-box; background-color:#fff; color:#000;}
				[class*="${selector.invite}"] input[type="submit"]{width:100%; height:30px; line-height:27px; font-size: 12px; font-weight: 900; text-align:center; background-color:#ddd; color:#000;}




				[class*="${selector.chat}"] [class*="${selector.scroll}"]{padding:2em; direction: rtl; display: flex; flex-direction: column; justify-content: flex-start; height: 100%; box-sizing:border-box; transform: rotate(180deg);}
				[class*="${selector.scroll}"]{display: block; overflow:hidden; overflow-y:scroll; box-sizing:border-box;}
				[class*="${selector.scroll}"]::-webkit-scrollbar {width: 5px;}
				[class*="${selector.scroll}"]::-webkit-scrollbar-thumb {background-color: rgba(255,255,255,0.5);}
				[class*="${selector.scroll}"]::-webkit-scrollbar-track {}

				[class*="${selector.checkbox}"]{display: block; opacity:0; width:0; height:0;}

				[class*="${selector.chat}"] [class*="${selector.talk}"]{position: relative; display: block; margin-bottom:2em; padding: 10px; text-align:right;}
				[class*="${selector.chat}"] [class*="${selector.talk}"] [class*="${selector.created_at}"]{display: block; margin-top:10px; width:100%; font-size:8px; text-align:right;}
				

				[class*="${selector.chat}"] [class*="${selector.talk}"] [class*="${selector.content}"]{display: block; margin-left:90px; border-radius:10px;}
				

				[class*="${selector.chat}"] [class*="${selector.talk}"] [class*="${selector.$background}"]{}

				[class*="${selector.chat}"] [class*="${selector.talk}"] [class*="${selector.message}"]{position: relative; overflow:hidden; display: inline-block; padding: 10px; border:1px solid #000; border-radius: 1em; text-align: left;}
				[class*="${selector.chat}"] [class*="${selector.talk}"] [class*="${selector.message}"]:after{content: ""; position: absolute; right: 0; bottom: 0; width: 100%; height: 100%; z-index:-1;}

				[class*="${selector.chat}"] [class*="${selector.talk}"] [class*="${selector.message}"] text{display: inline; line-height: 1.4; font-size: 1em; word-break: break-all; background-color:rgba(0,0,0,0.8); color:#fff;}
				

				[class*="${selector.chat}"] [class*="${selector.talk}"] [class*="${selector.message}"] deco{display: block; position: absolute; left: 0; top: 0; width: 100%; height: 100%; background-repeat: repeat; background-position: center; z-index:-1;}
				

				[class*="${selector.chat}"] [class*="${selector.talk}"][class*="${selector.$prompt}"]{text-align:left;}	

				[class*="${selector.chat}"] [class*="${selector.talk}"][class*="${selector.$prompt}"] [class*="${selector.message}"]{margin-left:0; margin-right:50px;}
				[class*="${selector.chat}"] [class*="${selector.talk}"][class*="${selector.$prompt}"] [class*="${selector.created_at}"]{text-align:left;}

				[class*="${selector.chat}"] [class*="${selector.talks}"]{position: relative; display: flex; flex-direction: column; justify-content: flex-end; width: 100%; height: 100%; z-index:1; box-sizing:border-box; transform: rotate(180deg);}

				[class*="${selector.chat}"] [class*="${selector.talks}"]:not(:empty) + [class*="${selector.start}"]{display:block; margin-bottom:3em; text-align:center;}

				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] [class*="${selector.message}"]{padding:0; border:0;}
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] [class*="${selector.content}"]{margin-left: 0; margin-right: 0;}
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] deco{background-image: none !important; background: transparent;}

				
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text{background:transparent; line-height:2;}
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text *{line-height:2;}
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text div{color:#fff;}
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text td{width:auto;}

				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text h1,
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text h2,
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text h3,
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text h4,
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text h5,
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text h6,
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text hr{margin:1em 0;}
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.label}"]{margin-bottom:0.5em;}


				[name="${selector.chat}"]{position: absolute; left: 0; right: 0; bottom: 0;}
				[name="${selector.chat}"] [type="text"]{padding: 0 7.5em 0 1em; width: 100%; height: 67px; background-color: #fff; color: #000; box-sizing: border-box;}
				[name="${selector.chat}"] [type="submit"]{position: absolute; right: 5em; bottom: 0; height: 67px; text-decoration: underline; color: #000;}

				[class*="${selector.label}"]{position: relative; display: block; margin-bottom: 1.5em; margin-left: 1em;}
				[class*="${selector.label}"]:after{content: ""; position: absolute; left: 0; top: 0; width: 100%; height: 100%; z-index:1;}
				[class*="${selector.left}"] [class*="${selector.label}"]:hover span{text-decoration: underline;}

				[class*="${selector.pages}"] [class*="${selector.label}"]{display:inline-block;}
				[class*="${selector.pages}"] [class*="${selector.label}"] u{text-decoration: underline; font-style: italic; font-size: 10px; vertical-align: top;}
				[class*="${selector.pages}"] [class*="${selector.label}"] i{position: absolute; background: red; border-radius: 50%; width: 14px; height: 14px; text-align: center; line-height: 13px; font-size: 10px; text-indent: -1px; margin: -5px 0 0 2px;}





				


				[id="${selector.right}"]:checked+[class*="${selector.area}"] [class="${selector.sender}"]{pointer-events: none;}

				[id="${selector.right}"]:checked+[class*="${selector.area}"] [id="${selector.address}"]:checked+[class="${selector.results}"] [class="${selector.sender}"]{bottom: 1em; right:1em;}

				[id="${selector.right}"]:checked+[class*="${selector.area}"] [class="${selector.sender}"] [for="${selector.address}"]{pointer-events: initial;}

				[class="${selector.sender}"]{position: absolute; left: 0; right: 0; bottom: 27px; margin: 0 auto; padding: 0 1em; max-width: 500px; z-index: 10; box-sizing:border-box; pointer-events: none;}
				[for="${selector.address}"]{padding: 5px 0; border-radius: 10px; line-height: 1.8; text-transform: capitalize; text-decoration: underline; color: #000; z-index: 10;}


				[id="${selector.file}"]{display:none;}
				[for="${selector.file}"]{position: absolute; right: 8em; bottom: 9px; overflow: hidden; padding: 7px 0; text-overflow: ellipsis; text-decoration:underline; color:#000;}
				[id="${selector.reset}"]{position: absolute; right: 3.5em; bottom: 10px; padding: 7px 0; text-transform: uppercase; text-decoration: underline; color: #000;}


				[class*="${selector.app}"][${selector.address}=""] [class*="${selector.sender}"],
				[class*="${selector.app}"][${selector.sender}=""] [class*="${selector.sender}"]{display:none;}

				[class*="${selector.app}"][${selector.sender}=""] [name="${selector.address}"]{display:block;}
				[class*="${selector.app}"][${selector.sender}=""] [name="${selector.prompt}"]{display:none;}
				[class*="${selector.app}"][${selector.sender}=""] [name="${selector.address}"] [type="submit"]{right: 1em;}


				[id="${selector.address}"]{display: none;}
				[id="${selector.address}"]:checked+[class="${selector.results}"] [name="${selector.address}"]{display:block;}
				[id="${selector.address}"]:checked+[class="${selector.results}"] [for="${selector.address}"]{position: absolute; left:initial; right: 14px; bottom: 17px; margin: 0 auto; padding: 0; text-decoration: none; border-radius: 15px; width: 36px; height: 36px; text-align: center; font-size: 0; font-family: 'Noto Color Emoji'; box-shadow: 0 0 4px #000; background: #000000d6; cursor: pointer; z-index: 100; pointer-events: initial;}
				[id="${selector.address}"]:checked+[class="${selector.results}"] [for="${selector.address}"]:after{content: "❌"; filter: sepia(100%) saturate(100%) grayscale(1); display: block; text-indent: 1.0px; line-height: 32px; font-size: 15px; background-size: 32px; background-position: center;}
				[id="${selector.address}"]:checked+[class="${selector.results}"] [name="${selector.prompt}"]{display:none;}

				[id="${selector.address}"]:checked+[class="${selector.results}"] [for="${selector.address}"]{}

				[name="${selector.address}"]{display: none; position: absolute; left: 2em; right: 2em; bottom: 0; margin: 0 auto; max-width: 500px; z-index:9;}
				[name="${selector.address}"] [type="submit"]{position: absolute; right: 4em; bottom: 2.9em; margin: 1em; height: 16px; color: #000; text-decoration: underline;}

				[name="${selector.sender}"]{display:block; width:100%; max-width: 500px; margin: 2em auto; border-radius: 1em; height: 67px; line-height: 1.3; padding:1em; background-color:#ddd; box-sizing: border-box; color:#000; box-shadow: 0 0 20px 0px #000;}



				[id="${selector.right}"]{display: none; pointer-events: initial;}

				[class*="${selector.app}"][${selector.address}=""] [id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.qrauth}"],
				[class*="${selector.app}"][${selector.address}=""] [id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.qrcode}"]{display:block;}
				[class*="${selector.app}"][${selector.address}=""] [id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.filters}"]
				[class*="${selector.app}"][${selector.address}=""] [id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.filters}"]{background-color: rgba(0, 0, 0, 0.5)}
				[class*="${selector.app}"][${selector.address}=""] [id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.filters}"] *{display:none;}

				[class*="${selector.app}"][${selector.address}=""] [id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.chat}"],
				[class*="${selector.app}"][${selector.address}=""] [id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.qrauth}"]+[class*="${selector.chat}"]{display:none;}

				


				[id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.chat}"]{display:block;}

				[id="${selector.right}"]:checked+[class*="${selector.area}"] [for="${selector.right}"]{transform:scale(1);}

				[id="${selector.right}"]:checked+[class*="${selector.area}"] [for="${selector.right}"]:after{content:"❌"; filter: sepia(100%) saturate(100%) grayscale(1);}


				[for="${selector.left}"]{}
				[id="${selector.left}"]{display: none; pointer-events: initial;}


				[class*="${selector.left}"]{overflow: hidden; position: absolute; left: 0; right: 0; bottom: 0; top: 40px; z-index: -1; opacity: 0; transition-duration: 0.3s;}
				[class*="${selector.left}"][class*="${selector.ocr}"]{background-color:#000;}
				[class*="${selector.left}"][class*="${selector.dom}"]{background-color:#fff;}

				[class*="${selector.left}"] [id="${selector.parse}"]{}

				[for="${selector.left}"][status="on"]{}
				[for="${selector.left}"][status="off"]{}

				[for="${selector.left}"] i{}

				[for="${selector.left}"][status="error"]{background-color:#9f9f9fd6;}

				[id="${selector.left}"]:checked+[class*="${selector.app}"] [class*="${selector.left}"]{display:block; pointer-events:initial; opacity:1;  z-index:99;}
				[id="${selector.left}"]:checked+[class*="${selector.app}"] [class*="${selector.left}"] *{pointer-events:initial;}
				[id="${selector.left}"]:checked+[class*="${selector.app}"] [class*="${selector.chat}"]{display:none;}


				[id="${selector.left}"]:checked+[class*="${selector.app}"] [for="${selector.left}"]{right:12px; transform:scale(1);}
				[id="${selector.left}"]:checked+[class*="${selector.app}"] [class*="${selector.area}"] [for="${selector.right}"]{display:none;}


				[id="${selector.left}"]:checked+[class*="${selector.app}"] [for="${selector.left}"]{box-shadow: 0 0 4px #000; background: #000000d6;}
				[id="${selector.left}"]:checked+[class*="${selector.app}"] [for="${selector.left}"]:before{display:none;}
				[id="${selector.left}"]:checked+[class*="${selector.app}"] [for="${selector.left}"]:after{content:"❌"; filter: sepia(100%) saturate(100%) grayscale(1); display: block; text-indent: 1.0px; line-height: 32px; font-size:15px; -webkit-text-stroke:0; transition-duration: 0.2s; background-size:32px; background-position:center;}



				[class*="${selector.ocr}"] [id="${selector.video}"]{position: absolute; left: 0; top: 0; width: 100%; height: 100%; object-fit:cover; z-index:1;}
				[class*="${selector.ocr}"] [id="${selector.canvas}"]{position: absolute; left: 0; top: 0; width: 100%; height: 100%; opacity:0; z-index:-1;}
				[class*="${selector.ocr}"] [id="${selector.photo}"]{position: absolute; left: 0; top: 0; width: 100%; height: 100%; object-fit:cover; z-index:2;}
				[class*="${selector.ocr}"] [id="${selector.parse}"]{position: absolute; left: 0; right: 0; bottom:5em; margin: 0 auto; width:50px; height:50px; border-radius:50%; background: #fff; opacity:0.7; z-index:3;}
				[class*="${selector.ocr}"] [id="${selector.parse}"]:after{content: ""; position: absolute; left: -1em; top:-1em; right:-1em; bottom:-1em; border:1px solid #fff; border-radius:50%; }

				[id="${selector.camera}"]{position: absolute; left: 1em; top: 1em; opacity:0.7; z-index:4;}


				
				[id="${selector.canvas}"]{}
				[id="${selector.photo}"]{}
				[id="${selector.video}"]{}
				[id="${selector.parse}"]{}

				[id="${selector.camera}"]{}


				[class*="${selector.menu}"]{padding: 0.4em 1em; border-radius: 10px; line-height: 1.8; text-transform: capitalize;}
				[class*="${selector.menu}"][id="${landing.home}"]{text-transform: inherit;}

				



				[class*="${landing.section}"]{display: block; padding: 8em 0;}
				[class*="${landing.section}"] [class*="${landing.title}"]{display: block; margin-bottom:15px; text-align: center;}
				[class*="${landing.section}"] [class*="${landing.title}"] span{font-size:30px; font-weight:900; line-height: 1.5; letter-spacing: 1px;}
				[class*="${landing.section}"] [class*="${landing.desc}"]{display: block; text-align: center;}
				[class*="${landing.section}"] [class*="${landing.desc}"] span{font-size:18px; line-height: 1.5;}
				

				[id="${landing.headline}"]{display: block; position: relative; margin: 7em 0; text-align: center; z-index: 0;}
				[id="${landing.headline}"] div{display:block; line-height:1.5; text-align: center;}
				[id="${landing.headline}"] h2{display: inline-block; margin-top: 0.1em; font-size: 3.3em; font-weight: 200;}

				[id="${landing.headline}"] strong{display: inline-block; position: relative; line-height: 1.3; font-size: 1em; font-weight: 900;}
				[id="${landing.headline}"] span{margin-top: 25px; font-size: 20px; display: block; text-align: center;}

				
				[class*="${landing.link}"]{display: block; margin: 20px auto 0; padding: 10px 0 12px 3px; max-width: 106px; border-radius: 14px; text-align: center; font-size: 10px; font-weight: 900; background-color: #235bf5; color: #fff; cursor: pointer;}


				[class*="${selector.app}"][${selector.address}=""] [id="${selector.scan}"]{display:none;}

				[id="${selector.scan}"]{position: absolute; right: 73px; bottom: 17px; margin: 0 auto; border-radius: 15px; width: 36px; height: 36px; text-align: center; font-size: 25px; background: #ffffffaa; cursor: pointer; z-index: 100; pointer-events: initial; transform: scale(1.3); transition: transform 0.2s;}
				[id="${selector.scan}"]:before{content: "📄"; position: absolute; left: 4px; top: 0; right: 0; text-align: center; text-decoration: line-through; line-height: 35px; filter: grayscale(1) invert(1); font-size: 23px;}
				[id="${selector.scan}"]:after{content: "⛶"; display: block; line-height: 30px; font-weight: 100; font-size: 47px; -webkit-text-stroke: 0.5px #ffffff; transition-duration: 0.2s; background-position: center; color: #000000;}


				[class*="${selector.app}"][class*="${selector.scanning}"] [id="${selector.scan}"]{pointer-events: none;}
				[class*="${selector.app}"][class*="${selector.scanning}"] [id="${selector.scan}"] i{position: absolute; left: 0; top: 0; width: 100%; height: 10px; background-color: rgba(45, 183, 183, 0.54); z-index: 1; transform: translateY(135%); animation: move 0.7s cubic-bezier(0.15, 0.44, 0.76, 0.64); animation-iteration-count: infinite;}

				@keyframes move {
					0%, 100% { transform: translateY(135%); }
					50% { transform: translateY(0%); }
					75% { transform: translateY(272%); }
				}

				@media (max-width: 740px) {
					[class*="${selector.prompt}"]{bottom:3em;}
				}

			`


		}else{

			ShadowStyle.innerHTML = `
				*{ margin: 0; padding: 0; border: 0; line-height: 1; text-decoration:none; list-style:none; font-style:normal; font-size:14px; font-family: 'Noto Sans KR', sans-serif; color: #000; }

				body{background:#000;}
				[class*="${selector.area}"] *,
				[class*="${selector.app}"]{all: initial; line-height:1em; font-size:14px; font-weight:500; font-family: sans-serif; line-height: 1; color:#fff;}
				[class*="${selector.app}"]{position: fixed; right: 0; top: 0; bottom: 0; width: 100%; z-index:100000000000000;}
				
				[class*="${selector.row}"]{display: table; width:100%; height:100%;}
				[class*="${selector.col}"]{display: table-cell; width:auto; height:100%;}

				[for="${selector.pip}"]{position:absolute; top: 0; right: 0; z-index:1;}

				[id="${landing.home}"]{display: block;}

				[id="${selector.sign.in}"]{display: none; font-size:12px;}
				[id="${selector.sign.out}"]{display: inline-block; font-size:12px;}

				[class*="${selector.app}"][${selector.address}=""] [class*="${selector.area}"] [id="${selector.sign.out}"]{display: none;}
				[class*="${selector.app}"][${selector.address}=""] [class*="${selector.area}"] [id="${selector.sign.in}"]{display: inline-block;}





				[id="${selector.header}"]{display: none;}
				[id="${selector.header}"]+[class*="${selector.header}"]{}
				[class*="${selector.header}"]{position:absolute; top: 0; left: 0; right: 0; padding-left: 60px; height:40px; z-index:8; background-color: #000; box-shadow: 0 0 10px 0px #000;}

				[class*="${selector.header}"] h1[class*="${selector.menu}"]{position:absolute; left:0.5em; top: 0; text-transform:inherit;}
				[class*="${selector.header}"] a[class*="${selector.menu}"]{position:absolute; right:0.5em; top: 2px;}

				[class*="${selector.center}"]{}

				[class*="${selector.left}"]{position:fixed; left:-265px; top: 40px; bottom: 0; left: 0; max-width:265px; width:265px; z-index: 9; transition-duration:0.3s;}
				[class*="${selector.left}"]::-webkit-scrollbar {height: 3px;}
				[class*="${selector.left}"]::-webkit-scrollbar-thumb {background-color: rgba(255,255,255,0.5);}
				[class*="${selector.left}"]::-webkit-scrollbar-track {}


				[id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.right}"]{right:0;}


				[id="${selector.left}"]:checked+[class*="${selector.app}"] [class*="${selector.left}"]{left:0;}
				[id="${selector.left}"]:checked+[class*="${selector.app}"] [class*="${selector.header}"]{}

				[class*="${selector.right}"]{position:fixed; top: 40px; bottom: 0; right: -400px; min-width: 400px; max-width: 400px; width: 400px; z-index:9; background-color:#111;}

				[class*="${selector.filters}"]{display:block; position: relative; vertical-align: top; background-color:#111;}
				[class*="${selector.filters}"] [class*="${selector.scroll}"]{display: flex; flex-flow: column; padding: 0 0 150px; height: 100%; box-sizing:border-box;}

				[id="${selector.system}"]{display:none;}
				[id="${selector.system}"]:checked+[class*="${selector.chat}"] [for="${selector.system}"]{opacity:1;}
				[id="${selector.system}"]:checked+[class*="${selector.chat}"] [for="${selector.system}"]:before{filter: none;}

				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.system}"]{display:none;}
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.system}"] deco{opacity:0.5;}
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"]:hover deco{opacity:1;}

				[id="${selector.system}"]:checked+[class*="${selector.chat}"] [class*="${selector.talk}"]{display:none;}
				[id="${selector.system}"]:checked+[class*="${selector.chat}"] [class*="${selector.talk}"][class*="${selector.system}"]{display:block;}
				[for="${selector.system}"]{position:absolute; top: -29px; right: 1em;opacity:0.5; cursor:pointer;}
				[for="${selector.system}"]:before{content:"✅"; margin-right: 5px; filter: sepia(100%) saturate(100%) grayscale(1);}
				



				[class*="${selector.pip}"]{display: block;}

				[name="${selector.interval}"]{height: 40px; line-height: 40px; appearance: base-select;}
				[name="${selector.interval}"] option{color:#000;}

				[id="${selector.interval}"]{position: relative; display: none; vertical-align: top;}
				[id="${selector.interval}"]:before{content: "/"; margin-left: 13px; margin-right: 13px; font-size: 12px; opacity:0.5;}

				[id="${selector.page}"]{position: relative; display: none; line-height: 40px; vertical-align: top; font-weight:bold; text-transform: capitalize;}
				[id="${selector.page}"]:before{content: "/"; margin-right: 13px; font-size: 12px; opacity:0.5;}
			
				[id="${selector.page}"]:not(:empty),
				[id="${selector.page}"]:not(:empty)+[id="${selector.interval}"]{display: inline-block;}


				[class*="${selector.profile}"]{position:absolute; bottom: 0.4em; left: 1em; right: 1em; z-index: 1;}
				[class*="${selector.profile}"]:after{content: ""; position: absolute; bottom: -1em; left: 0; right: 0; top: 1em; background-color: #111; z-index: -1;}
				[class*="${selector.profile}"] [class*="${selector.menu}"]{cursor:pointer;}
				
				[class*="${selector.profile}"] [class*="${selector.menu}"][id="${selector.sign.in}"]{}
				[class*="${selector.profile}"] [class*="${selector.menu}"][id="${selector.sign.out}"]{opacity:0.5;}


				[class*="${selector.profile}"] [class*="${selector.info}"]{position:relative; overflow:hidden; display: block; padding:0.7em; border-radius:1em; min-height: 40px; background-color:#000; box-shadow: 0 0 20px 0px #000; z-index:0;}

				[class*="${selector.favicon}"]{display: block; position: absolute; left:0; right:0; top: 0; bottom:0; background-repeat: repeat; background-position: center; z-index:-1; visibility:initial;}
				[class*="${selector.name}"]{position: relative; display: block; height:40px; line-height:40px;}

				[name="${selector.setting}"]{position:absolute; top: 0; left: 0; right: 0; bottom: 0; opacity:0; padding-right:50px; z-index:-1; background-color:#fff;}
				[name="${selector.setting}"] input[type="text"]{display: block; width:100%; height: 40px; box-sizing:border-box; color:#000;}
				[for="${selector.setting}"]{text-decoration: underline; background-color:rgba(0,0,0,1)}


				[class*="${selector.app}"][${selector.address}=""] [class="${selector.info}"],
				[class*="${selector.app}"][${selector.address}=""] [class="${selector.name}"]{display: none;}


				[class*="${selector.profile}"] [type="submit"]{position: absolute; top: 62px; right: 0; font-size: 12px; font-weight: 900; z-index: -1; color: #ffffff; text-decoration: underline; opacity: 0;}

				[class*="${selector.profile}"] [for="${selector.setting}"]{position:absolute; top: 10px; right: 0px; font-size:14px; z-index:2; background-color:#000;}

				[class*="${selector.name}"] strong{line-height: 39px; font-size:16px; background-color:#000;}

				[id="${selector.setting}"]:checked+[class*="${selector.profile}"] [type="submit"]{z-index:1; opacity:1;}

				[id="${selector.setting}"]:checked+[class*="${selector.profile}"] [class*="${selector.info}"]{overflow:initial; color:#000; background-color:#fff}
				[id="${selector.setting}"]:checked+[class*="${selector.profile}"] [class*="${selector.favicon}"]:after{border:1px solid #000;}

				[id="${selector.setting}"]:checked+[class*="${selector.profile}"] [name="${selector.setting}"]{opacity:1; z-index:5;}

				[id="${selector.setting}"]:checked+[class*="${selector.profile}"] [for="${selector.setting}"]{right: initial; left: -8px; top: -35px; font-size: 0; z-index: 7; background-color: rgba(0, 0, 0, 0);}
				[id="${selector.setting}"]:checked+[class*="${selector.profile}"] [for="${selector.setting}"]:after{content: 'Cancel'; font-size: 12px; font-weight: 700; text-decoration:underline;}


				[id="${selector.setting}"]:checked+[class*="${selector.profile}"] [class*="${selector.favicon}"]{display:none;}


				[id="${selector.setting}"]{display:none;}
				
				[id="${selector.setting}"]:checked+[class*="${selector.profile}"] [class*="${selector.menu}"]{position:relative; z-index:-1; opacity:0; pointer-events:none;}






				[class*="${selector.results}"]{position: relative; background-color: rgba(0, 0, 0, 0.5); z-index:2;}

				[class*="${selector.results}"] [class*="${selector.scroll}"]{width:100%; height:100%; padding:40px 9px 200px 1em; box-sizing:border-box; background:#222; background-image: radial-gradient(circle at center, #ffffff44 1px, transparent 0); background-size: 30px 30px; background-position: 0 0, 30px 30px; background-attachment: local; background-repeat: repeat;}
				[class*="${selector.results}"] [class*="${selector.scroll}"]:empty:after{content:"Empty"; position:absolute; left:0; right:0; top:50%; margin-top:-20px; font-size:2em; text-align:center; color:#fff; opacity:0.5;}
				[class*="${selector.results}"] [class*="${selector.scroll}"]:empty:after{}
				[class*="${selector.app}"][class*="${selector.loading}"] [class*="${selector.results}"] [class*="${selector.scroll}"]:empty:after{content:'Loading'; opacity:1;}
				

				[class*="${selector.results}"] [id*="more-"]{display:none;}
				
				[class*="${selector.results}"] [id*="more-"]:checked+[class*="${selector.result}"] [class*="${selector.info}"]{min-height:32px;}
				[class*="${selector.results}"] [id*="more-"]:checked+[class*="${selector.result}"] input{padding:0.5em 0.5em 0.5em 100px; background: #ddd; color: #000; font-weight:900; text-overflow: ellipsis; pointer-events:initial;}
				[class*="${selector.results}"] [id*="more-"]:checked+[class*="${selector.result}"] strong{padding: 0.9em 0; font-size: 11px; color:#000;}
				[class*="${selector.results}"] [id*="more-"]:checked+[class*="${selector.result}"] span i{top:9px; color:#000;}
				[class*="${selector.results}"] [id*="more-"]:checked+[class*="${selector.result}"] [class*="more-"]{display:block; margin-top:0;}
				[class*="${selector.results}"] [id*="more-"]:checked+[class*="${selector.result}"] [for*="more-"]:before{content:"▲ fold"; display: inline-block; vertical-align: top; font-size:14px; color:#000; text-decoration: underline;}
				[class*="${selector.results}"] [id*="more-"]:checked+[class*="${selector.result}"] [for*="more-"]{line-height:32px; font-size:0; background:#ffff00aa;}
				[class*="${selector.results}"] [id*="more-"][disabled]:checked+[class*="${selector.result}"] [for*="more-"]{background: #ddd;}
				[class*="${selector.results}"] [id*="more-"][disabled]:checked+[class*="${selector.result}"] [for*="more-"]:before{display:none;}


				

				[class*="${selector.result}"]{position: relative; overflow: hidden; display: block; margin: 2em auto 0; max-width: 500px; box-sizing: border-box;}

				[class*="${selector.result}"] a{position: relative; font-weight:bold; text-decoration:underline; z-index:2; cursor:pointer;}
				[class*="${selector.result}"] a *{cursor:pointer;}

				[class*="${selector.result}"] [class*="more-"]{margin-top:1em;}
			
				[class*="${selector.result}"] [for*="more-"]{display:block; padding-left:100px; cursor:pointer; text-decoration: underline;}
				[class*="${selector.result}"] [for*="more-"]:after{content:''; display:none; position:absolute; top: 0; left:0; right:0; bottom:0; transform:scale(100); z-index:1;}
				[class*="${selector.result}"] [for*="more-"]:before{content:"▼ "}

				[class*="${selector.results}"] [class*="more-"]{display:none;}

				[class*="${selector.results}"] [class*="${selector.talk}"]{display: block; padding:0.5em 1em; cursor:pointer;}
				[class*="${selector.results}"] [class*="${selector.talk}"] *{cursor:pointer;}
				[class*="${selector.results}"] [class*="${selector.talk}"]+[class*="${selector.talk}"]{margin-bottom:1em;}
				[class*="${selector.results}"] [class*="${selector.talk}"]>div{}
				[class*="${selector.results}"] [class*="${selector.talk}"] div{}

				[class*="${selector.result}"][class*="${selector.extend}"]{margin-top:0;}
				[class*="${selector.result}"][class*="${selector.extend}"] [class*="more-"]{display:block; margin-top:0;}
				[class*="${selector.result}"][class*="${selector.extend}"] [class*="${selector.created_at}"]{display:none;}
				[class*="${selector.result}"][class*="${selector.extend}"] input{padding:0.5em 0.5em 0.5em 100px; background: #ccc; color: #000; font-weight:900; text-overflow: ellipsis; pointer-events:initial;}
				[class*="${selector.result}"][class*="${selector.extend}"] strong{padding: 0.9em 0; font-size: 11px; color:#000;}
				[class*="${selector.result}"][class*="${selector.extend}"] span i{top:9px; color:#000;}

				[class*="${selector.result}"][class*="${selector.extend}"] [for*="more-"]{line-height:32px; font-size:0; background:#ccc;}


				[class*="${selector.result}"] [class*="${selector.info}"]{position:relative; display: block; min-height:17px; background: #00000050;}
				[class*="${selector.result}"] [class*="${selector.info}"] strong{position:absolute; top: 1px; left:10px; font-size:12px; letter-spacing: -1px; text-transform: capitalize; opacity:0.5; z-index:1;}
				[class*="${selector.result}"] [class*="${selector.info}"] input{padding-left:100px; width:100%; box-sizing:border-box; pointer-events:none;}
				[class*="${selector.result}"] [class*="${selector.info}"] span{position:relative; display: block;}
				[class*="${selector.result}"] [class*="${selector.info}"] span i{position:absolute; right:1.5em; top:3px; font-style:italic; pointer-events:none;}

				[class*="${selector.result}"] .created_at{background: #ffff00aa;}
				[class*="${selector.result}"] .created_at strong,
				[class*="${selector.result}"] .created_at label{color:#000;}

				[class*="${selector.xcroll}"]::-webkit-scrollbar {width: 5px;}
				[class*="${selector.xcroll}"]::-webkit-scrollbar-thumb {background-color: rgba(255,255,255,0.5);}
				[class*="${selector.xcroll}"]::-webkit-scrollbar-track {}

				
				[class*="${selector.host}"]{display: inline-block; vertical-align: top;}

				[name="${selector.host}"]{display: block; padding: 0 1em; width: 100%; height: 40px; line-height: 39px; font-size: 1em; font-weight: 900; box-sizing:border-box; background-color:transparent;}
				[name="${selector.host}"]:empty{display:none;}
				[name="${selector.host}"] option{color:#000;}


				[id="${selector.filters}"] [class*="${selector.filters}"]{min-width:265px; height:100%;}


				[class*="${selector.area}"]{position: absolute; right: 0; bottom: 0; max-width: 100%; width: 100%; height: 100%; z-index:0;}
				[class*="${selector.area}"]:after{content: ""; position: absolute; right: 0; bottom: 0; width: 100%; height: 100%; backdrop-filter: blur(0); background: rgba(0, 0, 0, 0); z-index: -1; transition-duration:0.3s;}

				[class*="${selector.qrcode}"]{display: none; position: absolute; left: 0; right: 0; bottom: 70px; margin: 10px auto 3px; padding: 1em; border-radius: 1em; max-width: 300px; box-sizing: border-box; background-color: #fff;}
				[class*="${selector.qrcode}"] canvas{display:none;}
				[class*="${selector.qrcode}"] img{display:block; width:100%; max-width:100%; min-width: 100%;}
				[class*="${selector.qrauth}"]{display: none; position: absolute; left: 0; right: 0; bottom: 17px; max-width: 300px; margin: 0 auto; border-radius: 1em; height: 36px; line-height: 36px; background-color: #fff; color: #000000; font-weight: 900; text-transform: uppercase; text-decoration: underline; text-align: center; cursor: pointer; box-shadow: 0 0 20px 0px #00000055;}


				[class*="${selector.chat}"]{padding-bottom:105px; width: 100%; height: 100vh; box-sizing: border-box;}
				[class*="${selector.chat}"]:after{content: ""; position: absolute; left: 36px; right: 30px; bottom: 60px; box-shadow: 0px 0 30px 10px #000; z-index: -1;}



				[class*="${selector.prompt}"]{position: absolute; left:0; right:0; bottom: 0; display: block; margin:1em; z-index:10;}


				[class*="${selector.app}"][${selector.address}=""] [id="${selector.interval}"],
				[class*="${selector.app}"][${selector.address}=""] [name="${selector.prompt}"]{display:none;}



				[name="${selector.prompt}"][style]{position:relative; height:320px;}
				[name="${selector.prompt}"][style]:after{content: ""; position: absolute; left: 0; right: 0; bottom: 0; box-shadow: 0 0 50px 50px #00000099; z-index: -1;}

				[name="${selector.prompt}"][style] [id="${selector.reset}"]{color:#fff;}
				[name="${selector.prompt}"][style] [for="${selector.file}"],
				[name="${selector.prompt}"][style]+[class*="${selector.sender}"],
				[name="${selector.prompt}"][style] textarea{display:none;}
				[name="${selector.prompt}"]{overflow: hidden; position: relative; display: block; margin:1em auto; max-width: 500px; height: 125px; border-radius: 10px; background-color: #fff; background-repeat: no-repeat; background-size: contain; background-position: center; box-shadow: 0 0 20px 0px #00000055; z-index:2;}
				[name="${selector.prompt}"] textarea{color: #000;}

				[name="${selector.context}"]{overflow: hidden; overflow-y: scroll; display: block; margin-bottom:45px; padding: 0.5em 1em 0; width: 100%; height: 80px; line-height: 1.2; white-space: pre-wrap; overflow-wrap: break-word; box-sizing: border-box; color: #000;}

				[name="${selector.context}"]::-webkit-scrollbar {width: 5px;}
				[name="${selector.context}"]::-webkit-scrollbar-thumb {background-color: rgba(255,255,255,0.5);}
				[name="${selector.context}"]::-webkit-scrollbar-track {}

				[for="${selector.submit}"]{overflow: hidden; position: absolute; right: 10px; bottom: 10px; border-radius: 50%; cursor:pointer; z-index: 1;}
				[for="${selector.submit}"] img{display:block; width: 25px; height: 25px; cursor:pointer;}
				[id="${selector.submit}"]{display:none;}

				[class*="${selector.app}"] [id="${selector.submit}"]:after{content:""; position:absolute; left:0; top:0; width:100%; height:100%; border-radius:50%; z-index: 0; background-color:#fff;}
				[class*="${selector.app}"] [id="${selector.submit}"]:before{content:""; position:absolute; left:0; top:0; right:0; bottom:0; margin:auto; width:50%; height:50%; z-index: 1; background-color:#000;}


				[class*="${selector.hidden}"]{position: absolute; right: 0; bottom:0; width:100%; height: 100%; z-index: -1;}

				[for="${selector.right}"]{position: absolute; right: 12px; bottom: 16px; margin: 0 auto; border-radius: 15px; width: 36px; height: 36px; text-align: center; font-size: 25px; font-family: 'Noto Color Emoji'; box-shadow: 0 0 4px #000; background: #000000d6; cursor: pointer; z-index: 100; transform:scale(1.3); transition-duration: 0.2s;}
				[for="${selector.right}"]:after{content:"✨"; display: block; text-indent: 1.0px; line-height: 32px; font-size: 15px; transition-duration: 0.2s; background-size:32px; background-position:center;}
				
				[class*="${selector.app}"][${selector.page}] [for="${selector.right}"]:after{filter: none;}
				[class*="${selector.app}"][${selector.page}=""] [for="${selector.right}"]:after{filter: sepia(100%) saturate(100%) grayscale(1); color:#ffff00aa;}
				



				[class*="${selector.pages}"] [class*="${selector.branch}"] input{display:none;}
				[class*="${selector.pages}"] [class*="${selector.children}"]{}

				[class*="${selector.pages}"] strong{display: block; margin-bottom: 1em; font-weight: 300; font-size: 10px; font-style: italic; font-family: arial; vertical-align: top;}
				[class*="${selector.pages}"] [class*="${selector.label}"] strong{display: table-caption; margin: 0.3em 0 0; width: 80px; line-height:1.2; font-size:10px; font-weight:300; font-style:italic; opacity:0.5;}

				[class*="${selector.branch}"] [class*="${selector.active}"]{filter :none;}
				[class*="${selector.branch}"] [class*="${selector.active}"]>span{text-decoration: underline; background-color:#fff;}
				[class*="${selector.branch}"] [class*="${selector.active}"]>span span{color:#000;}
				[class*="${selector.branch}"] [class*="${selector.active}"]>span:before{content: "▶"; position: absolute; left: -2em; top: 1px; color: #fff; font-size: 6px;}
				[class*="${selector.branch}"] [class*="${selector.active}"]>span u{color:#000;}
				
				[class*="${selector.branch}"] [class*="${selector.label}"]{width: 80px;}
				[class*="${selector.branch}"] [class*="${selector.label}"] span>span:first-child{display:none;}
				[class*="${selector.branch}"] [class*="${selector.label}"]:last-child span>span:first-child{display:inline;}
				[class*="${selector.branch}"] [class*="${selector.label}"] + [class*="${selector.child}"] span>span{display:inline;}




				[class*="${selector.filters}"] label span{position: relative; display: inline-block; vertical-align:top; text-transform: capitalize; white-space: nowrap; font-weight: 300;}
				[class*="${selector.filters}"] label:hover>span{background-color:#fff; color:#000; text-decoration: underline;}
				[class*="${selector.filters}"] label:hover span span{color:#000; text-decoration: underline;}
				[class*="${selector.filters}"] label:hover span u{color:#000;}

				[class*="${selector.pages}"] [class*="${selector.children}"]>label>span{font-family: arial;}

				[class*="${selector.pages}"]{position: relative; display: block; margin: 1em 1em 0;}
				[class*="${selector.pages}"] [class*="${selector.parent}"]{overflow:initial;}
				[class*="${selector.pages}"]>[class*="${selector.branch}"]>li[class*="${selector.parent}"]{border-bottom: 1px dashed #434343; margin-bottom: 1.5em;}
				

				[class*="${selector.paginate}"]{margin-top:1em; display:block; text-align:center;}
			

				[class*="${selector.membership}"],
				[id="${selector.membership}"]{display:none;}
				[id="${selector.membership}"]:checked +[id="${selector.filters}"] [class*="${selector.membership}"]{display:block;}
				[id="${selector.membership}"]:checked +[id="${selector.filters}"] [for="${selector.membership}"]{font-size:0;}
				[id="${selector.membership}"]:checked +[id="${selector.filters}"] [for="${selector.membership}"]:after{content:'back'; font-size:10px; text-decoration: underline;}
				[for="${selector.membership}"]{position: absolute; left: 60px; top: -1px; font-size: 10px; font-weight: 300; text-decoration: underline;}

				[class*="${selector.users}"]{display: block; margin: 2em 1em 5em;}
				[class*="${selector.users}"] [class*="${selector.branch}"] [class*="${selector.label}"]{min-width:80px; width:auto;}


				[name="${selector.team}"]{display:none;}

				[id*="team-"]:checked+[class*="${selector.parent}"] [class*="team-"] [name="${selector.team}"]{display:block;}

				[team-id]{position: relative; display: block;}
				
				[team-id]>strong{float:left; margin-bottom: 0.5em; font-size:10px; font-weight:300; font-style: italic;}
				[team-id]>label{display:inline-block; font-size:10px; margin-bottom: 0; margin-left: 9px; padding:0; text-decoration:underline; vertical-align: top;}
				

				[user-id] [class*="${selector.label}"] i,
				[team-id] [class*="${selector.label}"] i{margin-left:5px; font-size:9px; letter-spacing:-0.5px; vertical-align: top;}



				[class*="${selector.parent}"]{position: relative; display: block; overflow:hidden;}
				[class*="${selector.child}"]{position: relative; display: inline-block; float: left; clear: both;}

				[id="${selector.invite}"]{display:none;}
				[id="${selector.invite}"]:checked+[class*="${selector.filters}"] [class*="${selector.users}"] label[class*="${selector.label}"][class*="user-"]{display:none;}

				[id="${selector.invite}"]:checked+[class*="${selector.filters}"] [class*="${selector.users}"] [class*="${selector.invite}"]{display:block;}

				[id="${selector.invite}"]:checked+[class*="${selector.filters}"] [class*="${selector.users}"] [for="${selector.invite}"]{position: absolute; bottom: -35px; left: 0; right: 0; font-size: 0; height: 30px;}
				[id="${selector.invite}"]:checked+[class*="${selector.filters}"] [class*="${selector.users}"] [for="${selector.invite}"]:after{content:"cancel"; display:block; font-size:14px; line-height:2; text-align:center;}


				label[for="${selector.invite}"]{margin-top: 1em; text-decoration: underline; font-size: 10px; text-indent: 3px;}

				[class*="${selector.invite}"]{overflow:hidden; position: relative; display: none; margin-top: 1em; border:1px solid rgba(255,255,255,0.1); border-radius:1em;}
				[class*="${selector.invite}"] input[type="email"]{width:100%; height:30px; padding:0 4px; box-sizing:border-box; background-color:#fff; color:#000;}
				[class*="${selector.invite}"] input[type="submit"]{width:100%; height:30px; line-height:27px; font-size: 12px; font-weight: 900; text-align:center; background-color:#ddd; color:#000;}




				[class*="${selector.chat}"] [class*="${selector.scroll}"]{padding:2em; direction: rtl; display: flex; flex-direction: column; justify-content: flex-start; height: 100%; box-sizing:border-box; transform: rotate(180deg);}
				[class*="${selector.scroll}"]{display: block; overflow:hidden; overflow-y:scroll; box-sizing:border-box;}
				[class*="${selector.scroll}"]::-webkit-scrollbar {width: 5px;}
				[class*="${selector.scroll}"]::-webkit-scrollbar-thumb {background-color: rgba(255,255,255,0.5);}
				[class*="${selector.scroll}"]::-webkit-scrollbar-track {}

				[class*="${selector.checkbox}"]{display: block; opacity:0; width:0; height:0;}

				[class*="${selector.chat}"] [class*="${selector.talk}"]{position: relative; display: block; margin-bottom:2em; padding: 10px; text-align:right;}
				[class*="${selector.chat}"] [class*="${selector.talk}"] [class*="${selector.created_at}"]{display: block; margin-top:10px; width:100%; font-size:8px; text-align:right;}
				

				[class*="${selector.chat}"] [class*="${selector.talk}"] [class*="${selector.content}"]{display: block; margin-left:90px; border-radius:10px;}
				

				[class*="${selector.chat}"] [class*="${selector.talk}"] [class*="${selector.$background}"]{}

				[class*="${selector.chat}"] [class*="${selector.talk}"] [class*="${selector.message}"]{position: relative; overflow:hidden; display: inline-block; padding: 10px; border:1px solid #000; border-radius: 1em; text-align: left;}
				[class*="${selector.chat}"] [class*="${selector.talk}"] [class*="${selector.message}"]:after{content: ""; position: absolute; right: 0; bottom: 0; width: 100%; height: 100%; z-index:-1;}

				[class*="${selector.chat}"] [class*="${selector.talk}"] [class*="${selector.message}"] text{display: inline; line-height: 1.4; font-size: 1em; word-break: break-all; background-color:rgba(0,0,0,0.8); color:#fff;}
				

				[class*="${selector.chat}"] [class*="${selector.talk}"] [class*="${selector.message}"] deco{display: block; position: absolute; left: 0; top: 0; width: 100%; height: 100%; background-repeat: repeat; background-position: center; z-index:-1;}
				

				[class*="${selector.chat}"] [class*="${selector.talk}"][class*="${selector.$prompt}"]{text-align:left;}	

				[class*="${selector.chat}"] [class*="${selector.talk}"][class*="${selector.$prompt}"] [class*="${selector.message}"]{margin-left:0; margin-right:50px;}
				[class*="${selector.chat}"] [class*="${selector.talk}"][class*="${selector.$prompt}"] [class*="${selector.created_at}"]{text-align:left;}

				[class*="${selector.chat}"] [class*="${selector.talks}"]{position: relative; display: flex; flex-direction: column; justify-content: flex-end; width: 100%; height: 100%; z-index:1; box-sizing:border-box; transform: rotate(180deg);}

				[class*="${selector.chat}"] [class*="${selector.talks}"]:not(:empty) + [class*="${selector.start}"]{display:block; margin-bottom:3em; text-align:center;}

				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] [class*="${selector.message}"]{padding:0; border:0;}
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] [class*="${selector.content}"]{margin-left: 0; margin-right: 0; border:0;}
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] deco{background-image: none !important; background: transparent;}

				
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text{background:transparent; line-height:2;}
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text *{line-height:2;}
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text div{color:#fff;}
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text td{width:auto;}

				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text h1,
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text h2,
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text h3,
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text h4,
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text h5,
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text h6,
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.talk}"][class*="${selector.markdown}"] text hr{margin:1em 0;}
				[class*="${selector.chat}"] [class*="${selector.talks}"] [class*="${selector.label}"]{margin-bottom:0.5em;}


				[name="${selector.chat}"]{position: absolute; left: 0; right: 0; bottom: 0;}
				[name="${selector.chat}"] [type="text"]{padding: 0 7.5em 0 1em; width: 100%; height: 67px; background-color: #fff; color: #000; box-sizing: border-box;}
				[name="${selector.chat}"] [type="submit"]{position: absolute; right: 5em; bottom: 0; height: 67px; text-decoration: underline; color: #000;}

				[class*="${selector.label}"]{position: relative; display: block; margin-bottom: 1.5em; margin-left: 1em;}
				[class*="${selector.label}"]:after{content: ""; position: absolute; left: 0; top: 0; width: 100%; height: 100%; z-index:1;}
				[class*="${selector.left}"] [class*="${selector.label}"]:hover span{text-decoration: underline;}

				[class*="${selector.pages}"] [class*="${selector.label}"]{display:inline-block;}
				[class*="${selector.pages}"] [class*="${selector.label}"] u{text-decoration: underline; font-style: italic; font-size: 10px; vertical-align: top;}
				[class*="${selector.pages}"] [class*="${selector.label}"] i{position: absolute; background: red; border-radius: 50%; width: 14px; height: 14px; text-align: center; line-height: 13px; font-size: 10px; text-indent: -1px; margin: -5px 0 0 2px;}



				
				[class="${selector.sender}"]{position: absolute; left: 0; right: 0; bottom: 24px; margin: 0 auto; padding: 0 1em; max-width: 500px; box-sizing: border-box; z-index: 10; pointer-events: none;}
				[for="${selector.address}"]{padding: 5px 0; line-height: 1.8; text-transform: capitalize; text-decoration: underline; color: #000; z-index: 10;}


				[id="${selector.file}"]{display:none;}
				[for="${selector.file}"]{position: absolute; right: 8em; bottom: 9px; overflow: hidden; padding: 7px 0; text-overflow: ellipsis; text-decoration:underline; color:#000;}
				[id="${selector.reset}"]{position: absolute; right: 3.5em; bottom: 10px; padding: 6px 0; text-transform: uppercase; text-decoration: underline; color: #000;}

				[class*="${selector.app}"][${selector.address}=""] [class*="${selector.sender}"],
				[class*="${selector.app}"][${selector.sender}=""] [class*="${selector.sender}"]{display:none;}

				[class*="${selector.app}"][${selector.sender}=""] [name="${selector.address}"]{display:block;}
				[class*="${selector.app}"][${selector.sender}=""] [name="${selector.prompt}"]{display:none;}
				[class*="${selector.app}"][${selector.sender}=""] [name="${selector.address}"] [type="submit"]{right: 1em;}


				[id="${selector.address}"]{display: none;}
				[id="${selector.address}"]:checked+[class="${selector.results}"] [name="${selector.address}"]{display:block;}
				[id="${selector.address}"]:checked+[class="${selector.results}"] [for="${selector.address}"]{position: absolute; left:initial; right: 12px; bottom: 17px; margin: 0 auto; padding: 0; text-decoration: none; border-radius: 15px; width: 36px; height: 36px; text-align: center; font-size: 0; font-family: 'Noto Color Emoji'; box-shadow: 0 0 4px #000; background: #000000d6; cursor: pointer; z-index: 100; pointer-events: initial;}
				[id="${selector.address}"]:checked+[class="${selector.results}"] [for="${selector.address}"]:after{content: "❌"; filter: sepia(100%) saturate(100%) grayscale(1); display: block; text-indent: 1.0px; line-height: 32px; font-size: 15px; background-size: 32px; background-position: center;}
				[id="${selector.address}"]:checked+[class="${selector.results}"] [name="${selector.prompt}"]{display:none;}

				[id="${selector.address}"]:checked+[class="${selector.results}"] [for="${selector.address}"]{}

				[name="${selector.address}"]{display: none; position: absolute; left: 1em; right: 1em; bottom: 0; margin: 2.9em auto; max-width: 500px; z-index: 9;}
				[name="${selector.address}"] [type="submit"]{position: absolute; right: 4em; bottom: 1em; margin: 1em; height: 16px; color: #000; text-decoration: underline;}

				[name="${selector.sender}"]{display:block; width:100%; height: 67px; line-height: 1.3; padding:1em; border-radius: 1em; background-color:#ddd; box-sizing: border-box; color:#000; box-shadow: 0 0 20px 0px #000;}




				[id="${selector.right}"]{display: none; pointer-events: initial;}

				[class*="${selector.app}"][${selector.address}=""] [id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.qrauth}"],
				[class*="${selector.app}"][${selector.address}=""] [id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.qrcode}"] img,
				[class*="${selector.app}"][${selector.address}=""] [id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.qrcode}"]{display:block; opacity:1; visibility:initial;}

				[class*="${selector.app}"][${selector.address}=""] [id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.chat}"],
				[class*="${selector.app}"][${selector.address}=""] [id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.qrauth}"]+[class*="${selector.chat}"]{display:none;}

				

				[id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.chat}"]{display:block;}
				[id="${selector.right}"]:checked+[class*="${selector.area}"] [for="${selector.right}"]{transform:scale(1);}

				[id="${selector.right}"]:checked+[class*="${selector.area}"] [for="${selector.right}"]:after{content:"❌"; filter: sepia(100%) saturate(100%) grayscale(1);}


				[for="${selector.left}"]{}
				[id="${selector.left}"]{display: none; pointer-events: initial;}


				[class*="${selector.left}"]{overflow:hidden; display: none; position:absolute; left:0; right:0; bottom:0; top:40px; z-index:-1; opacity:0; transition-duration:0.3s; transition-delay:0.3s;}
				[class*="${selector.left}"][class*="${selector.ocr}"]{background-color:#000;}
				[class*="${selector.left}"][class*="${selector.dom}"]{background-color:#fff;}

				[class*="${selector.left}"] [id="${selector.parse}"]{}

				[for="${selector.left}"][status="on"]{}
				[for="${selector.left}"][status="off"]{}

				[for="${selector.left}"] i{}

				[for="${selector.left}"][status="error"]{background-color:#9f9f9fd6;}

				[id="${selector.left}"]:checked+[class*="${selector.app}"] [class*="${selector.left}"]{display:block; pointer-events:initial; opacity:1;  z-index:7;}
				[id="${selector.left}"]:checked+[class*="${selector.app}"] [class*="${selector.left}"] *{pointer-events:initial;}
				
				[id="${selector.left}"]:checked+[class*="${selector.app}"] [id="${selector.scan}"]{display:none;}


				[id="${selector.left}"]:checked+[class*="${selector.app}"] [for="${selector.left}"]{}
				[id="${selector.left}"]:checked+[class*="${selector.app}"][class*="${selector.mobile}"] [class*="${selector.area}"] [for="${selector.right}"]{display:none;}

				[id="${selector.left}"]:checked+[class*="${selector.app}"] [class*="${selector.area}"] [class*="${selector.center}"] [class*="${selector.scroll}"]:after,
				[class*="${selector.app}"] [id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.center}"] [class*="${selector.scroll}"]:after{content:""; position:absolute; left:0; right:0; top:0; bottom: 0; z-index:10; background-color:rgba(0,0,0,0.8)}

				[id="${selector.left}"]:checked+[class*="${selector.app}"] [class*="${selector.area}"] [class*="${selector.center}"] [class*="${selector.scroll}"],
				[class*="${selector.app}"] [id="${selector.right}"]:checked+[class*="${selector.area}"] [class*="${selector.center}"] [class*="${selector.scroll}"]{overflow:hidden;}

				[id="${selector.left}"]:checked+[class*="${selector.app}"] [for="${selector.left}"]{}

				[for="${selector.left}"]{position: fixed; left: 1em; top: 0; margin: 0; line-height: 39px; height: 40px; cursor: pointer; z-index: 100;}
				[for="${selector.left}"]:before{content: "Menu"; color:#fff;}
				[id="${selector.left}"]:checked+[class*="${selector.app}"] [for="${selector.left}"]:before{content:'Close'}
				


				[class*="${selector.app}"] [class*="${selector.scan}"]{display:none; }


				[class*="${selector.app}"][class*="${selector.scan}"] [for="${selector.right}"],
				[class*="${selector.app}"][class*="${selector.scan}"] [for="${selector.left}"],
				[class*="${selector.app}"][class*="${selector.scan}"] [class="${selector.header}"],
				[class*="${selector.app}"][class*="${selector.scan}"] [class="${selector.results}"]{display:none;}

				[class*="${selector.app}"][class*="${selector.scan}"] [class*="${selector.scan}"]{display:block;}


				[class*="${selector.app}"] [id="${selector.right}"]:checked+[class*="${selector.area}"] [id="${selector.scan}"]{display:none;}


				[class*="${selector.app}"][class*="${selector.scan}"] [id="${selector.scan}"]{right: 12px; bottom: 16px; margin: 0 auto; border-radius: 15px; width: 36px; height: 36px; text-align: center; font-size: 25px; font-family: 'Noto Color Emoji'; box-shadow: 0 0 4px #000; background: #000000d6; cursor: pointer; z-index: 100; transform: scale(1);}
				[class*="${selector.app}"][class*="${selector.scan}"] [id="${selector.scan}"] i,
				[class*="${selector.app}"][class*="${selector.scan}"] [id="${selector.scan}"]:before{display:none;}
				[class*="${selector.app}"][class*="${selector.scan}"] [id="${selector.scan}"]:after{content: "❌"; filter: sepia(100%) saturate(100%) grayscale(1); display: block; text-indent: 1.0px; line-height: 32px; font-size: 15px;}





				[class*="${selector.ocr}"] [id="${selector.video}"]{position: absolute; left: 0; top: 0; width: 100%; height: 100%; object-fit:cover; background: #222; z-index:1;}
				[class*="${selector.ocr}"] [id="${selector.canvas}"]{position: absolute; left: 0; top: 0; width: 100%; height: 100%; opacity:0; z-index:-1;}
				[class*="${selector.ocr}"] [id="${selector.photo}"]{position: absolute; left: 0; top: 0; width: 100%; height: 100%; object-fit:cover; z-index:2;}
				[class*="${selector.ocr}"] [id="${selector.parse}"]{position: absolute; left: 0; right: 0; bottom:5em; margin: 0 auto; width:50px; height:50px; border-radius:50%; background: #fff; opacity:0.7; z-index:3;}
				[class*="${selector.ocr}"] [id="${selector.parse}"]:after{content: ""; position: absolute; left: -1em; top:-1em; right:-1em; bottom:-1em; border:1px solid #fff; border-radius:50%; }

				[id="${selector.camera}"]{position: absolute; left: 1em; top: 1em; opacity:0.7; z-index:4;}


				
				[id="${selector.canvas}"]{}
				[id="${selector.photo}"]{}
				[id="${selector.video}"]{}
				[id="${selector.parse}"]{}

				[id="${selector.camera}"]{}


				[class*="${selector.menu}"]{padding: 0.4em 1em; border-radius: 10px; line-height: 1.8; text-transform: capitalize;}
				[class*="${selector.menu}"][id="${landing.home}"]{text-transform: inherit;}

				



				[class*="${landing.section}"]{display: block; padding: 8em 0;}
				[class*="${landing.section}"] [class*="${landing.title}"]{display: block; margin-bottom:15px; text-align: center;}
				[class*="${landing.section}"] [class*="${landing.title}"] span{font-size:30px; font-weight:900; line-height: 1.5; letter-spacing: 1px;}
				[class*="${landing.section}"] [class*="${landing.desc}"]{display: block; text-align: center;}
				[class*="${landing.section}"] [class*="${landing.desc}"] span{font-size:18px; line-height: 1.5;}
				

				[id="${landing.headline}"]{display: block; position: relative; margin: 7em 0; text-align: center; z-index: 0;}
				[id="${landing.headline}"] div{display:block; line-height:1.5; text-align: center;}
				[id="${landing.headline}"] h2{display: inline-block; margin-top: 0.1em; font-size: 3.3em; font-weight: 200;}

				[id="${landing.headline}"] strong{display: inline-block; position: relative; line-height: 1.3; font-size: 1em; font-weight: 900;}
				[id="${landing.headline}"] span{margin-top: 25px; font-size: 20px; display: block; text-align: center;}

				
				[class*="${landing.link}"]{display: block; margin: 20px auto 0; padding: 10px 0 12px 3px; max-width: 106px; border-radius: 14px; text-align: center; font-size: 10px; font-weight: 900; background-color: #235bf5; color: #fff; cursor: pointer;}



				[class*="${selector.app}"][${selector.address}=""] [id="${selector.scan}"]{display:none;}

				[id="${selector.scan}"]{position: absolute; right: 73px; bottom: 17px; margin: 0 auto; border-radius: 15px; width: 36px; height: 36px; text-align: center; font-size: 25px; background: #ffffffaa; cursor: pointer; z-index: 100; pointer-events: initial; transform: scale(1.3);}
				[id="${selector.scan}"]:before{content: "📄"; position: absolute; left: 0; top: 0; right: 0; text-align: center; text-decoration: line-through; line-height: 35px; filter: grayscale(1) invert(1); font-size: 23px;}
				[id="${selector.scan}"]:after{content: "⛶"; display: block; text-indent: -2px; line-height: 30px; font-weight: 100; font-size: 47px; -webkit-text-stroke: 0.5px #ffffff; transition-duration: 0.2s; background-position: center; color: #000000;}


				[class*="${selector.app}"][class*="${selector.scanning}"] [id="${selector.scan}"]{pointer-events: none;}
				[class*="${selector.app}"][class*="${selector.scanning}"] [id="${selector.scan}"] i{position: absolute; left: 0; top: 0; width: 100%; height: 10px; background-color: rgba(45, 183, 183, 0.54); z-index: 1; transform: translateY(135%); animation: move 0.7s cubic-bezier(0.15, 0.44, 0.76, 0.64); animation-iteration-count: infinite;}

				@keyframes move {
					0%, 100% { transform: translateY(135%); }
					50% { transform: translateY(0%); }
					75% { transform: translateY(272%); }
				}


				@media (max-width: 740px) {
					[class*="${selector.prompt}"]{bottom:3em;}
				}

			`
		}

			


		var logo = `<img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZlcnNpb249IjEuMiIgdmlld0JveD0iMC45NTQ3Mzg2MTY5NDMzNTk0IDAuOTU0NzQyNDMxNjQwNjI1IDc4LjA5MDUxNTEzNjcxODc1IDc4LjA5MDUxNTEzNjcxODc1IiB3aWR0aD0iNTAwIiBoZWlnaHQ9IjUwMCI+Cgk8dGl0bGU+TDhiVVZqdENyR0NHWktSeDBFejU4Qm5nYklEekcyeHh1WHZRcFRmcHNzV0hDSkFJRDVVeGVfN3Q0Zi12NlBhSS1IZ2FkU0JqYkJMTVN6WFBfVHlkMnc8L3RpdGxlPgoJPHN0eWxlPgoJCS5zMCB7IGZpbGw6ICMwMDAwMDAgfSAKCQkuczEgeyBmaWxsOiAjZmZmZmZmIH0gCgk8L3N0eWxlPgoJPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGFzcz0iczAiIGQ9Im0yLjMgMjkuOWM1LjYtMjAuOCAyNy0zMy4yIDQ3LjgtMjcuNiAyMC44IDUuNiAzMy4yIDI3IDI3LjYgNDcuOC01LjYgMjAuOC0yNyAzMy4yLTQ3LjggMjcuNi0yMC44LTUuNi0zMy4yLTI3LTI3LjYtNDcuOHoiLz4KCTxwYXRoIGNsYXNzPSJzMSIgZD0ibTc2LjcgNDkuOGMyLjYtOS43IDEuMi0yMC4xLTMuOC0yOC44LTUtOC43LTEzLjMtMTUuMS0yMy4xLTE3LjctOS43LTIuNi0yMC4xLTEuMi0yOC44IDMuOC04LjcgNS0xNS4xIDEzLjMtMTcuNyAyMy4xLTEuMyA0LjgtMC42IDEwIDEuOSAxNC40IDIuNSA0LjMgNi43IDcuNSAxMS41IDguOCA0LjkgMS4zIDEwLjEgMC43IDE0LjQtMS45IDQuNC0yLjUgNy42LTYuNiA4LjktMTEuNSAxLjMtNC45IDQuNS05IDguOS0xMS41IDQuMy0yLjYgOS41LTMuMiAxNC40LTEuOSA0LjggMS4zIDkgNC41IDExLjUgOC44IDIuNSA0LjQgMy4yIDkuNiAxLjkgMTQuNHoiLz4KPC9zdmc+" />`

		/*
			메일 레아아웃

				Column 1
					selector.filters

					- 쇼핑몰 선택 Select tagName // 현재 활성화 쇼핑몰

					- unread

					- 메모 메뉴
					- 과제 메뉴

					- 상품 메뉴
					- 주문 메뉴 
					- 배송 메뉴 

				Column 2
					selector.results

				Column 3
					selector.result
					
					상세 or 결과 페이지
		*/
		document.head.appendChild(ExtendStyle)

		shadowRoot.appendChild(ShadowStyle)
		shadowRoot.appendChild($app)
		// document.body.appendChild($app)


		// 초기화 chrlghk
	
		// try{
		// 	await Clear['pages']()
		// }catch(err){

		// }

		// try{
		// 	await Clear['items']()
		// }catch(err){

		// }

		// try{
		// 	await Clear['crons']()
		// }catch(err){

		// }

		// try{
		// 	await Clear['users']()
		// }catch(err){

		// }

		// await app.storage.set({'cookies' : {}})

		// return
		

		

		

		

		

		var scrollTemp = {}


		var page

		var timeout = {
			ms : 1000,
			clear : async function(){
				var { cookies } = await app.storage.get('cookies')

				if(cookies){
					clearTimeout(window[cookies.hash])
					window[cookies.hash] = null
				}
			},
			fn : async function(event){
				timeout.clear()

				console.log('app.block',app.block);

				try{
					var { cookies } = await app.storage.get('cookies')

					var _crons = []

					try{
						var crons = await Select['crons']()

						if(crons.length){
							for(var c = 0; c < crons.length; c++){
								var cron = crons[c]

								_crons.push(cron.id)
							}
						}
					}catch(err){
						console.log('err',err);
					}

						

					var { results, session } = await app.fetch({
						url : reqUrl( cookies, app.filters, { href : window.location.href, crons : JSON.stringify(_crons) } ),
						method: "GET",
						headers: {
							"Content-Type": "application/json"
						}
					})





					console.log('timeout results',results);

					console.log('session',session,cookies);

					if(session){
						var $sender = $app.querySelector(`[name="${selector.sender}"]`)

						if(!cookies.address){
							if(session.address){
								retryCount = 0
								
								timeout.ms = 2000

								var $context = $app.querySelector(`[name="${selector.context}"]`)

								$context.setAttribute('placeholder', placeholder.prompt())

								$context.placeholder = session.hello

								$app.setAttribute(selector.address, session.address ? session.address : "")
								$app.setAttribute(selector.sender, session.sender ? hashId(session.sender) : "")

								$sender.value = session.sender ? session.sender : ''

								var seed = session.address ? session.address : Ethers.ZeroAddress;

								var canvas = blockies.create({seed: seed.toLowerCase()});
								
								var base64 = canvas.toDataURL();

								var $profile = $app.querySelector(`[class*="${selector.profile}"]`)

								console.log('$profile',$profile);

								$profile.querySelector(`[name="${selector.setting}"] [type="text"]`).value = session.name ? session.name : ""

								$profile.querySelector(`[class*="${selector.name}"] strong`).textContent = session.name ? session.name : ""

								$profile.querySelector(`[class*="${selector.favicon}"]`).setAttribute('style',`background-image:url(${base64})`)

							}else if(retryCount){
								var $qrauth = $app.querySelector(`[class*="${selector.qrauth}"]`)

								var dots = ""

								for(var d = 0; d < retryCount; d++){
									dots += "."
								}

								retryCount += 1

								if(retryCount > 3){
									retryCount = 1
								}

								$qrauth.textContent = `${dots}`	
							}else{
								$app.setAttribute(selector.address, "")


								var $qrcode = $app.querySelector(`[class*="${selector.qrcode}"]`)

								if(!$qrcode.querySelector(`img`)){
									new QRCode($qrcode, {
										text: "mailto:"+encodeURIComponent(cookies.hash+".logis.center@oauth.email"),
										width: 300,
										height: 300,
										colorDark : "#000000",
										colorLight : "#ffffff",
										correctLevel : QRCode.CorrectLevel.H
									})
								}
							}

						}else{
							if(!$sender.value && session.sender){
								$sender.value = session.sender
							}
						}

						try{
							cookies = session
						}catch(err){
							console.log('err',err);
						}


						var footprint = new URL(window.location.href.toLowerCase())

						var origin = footprint.origin;

						var pathname = footprint.pathname;

						var search = footprint.search;

						var link = pathname + search

						var scanId = hashId(cookies.cc+link)

						var pageId = hashId(cookies.team+cookies.cc+pathname)

						var itemId = hashId(cookies.team+cookies.cc+link)

						var isDetail = false

						try{
							if(Object.keys(app.filters.page).length){
								var pageData = app.filters.page.data

								if(!pageData.item){
									isDetail = true
								}

								if(pageData.origin && pageData.link){
									footprint = new URL(pageData.origin+pageData.link)

									pageId = hashId(cookies.team+(isDetail ? cookies.cc.toUpperCase() : cookies.cc)+footprint.pathname)

									link = footprint.pathname + footprint.search

									itemId = hashId(cookies.team+cookies.cc+link)
								}
							}
						}catch(err){
							console.log('page err',err)
						}


						if(session.address){
							if(cookies.address){
								if(session.address != cookies.address){
									// 세션 "미일치"시 제거
									try{
										await Clear['pages']()
									}catch(err){

									}

									try{
										await Clear['items']()
									}catch(err){

									}

									try{
										await Clear['crons']()
									}catch(err){

									}

									try{
										await Clear['users']()
									}catch(err){

									}

								}
							}

							var userDiff = false

							var pageDiff = isDiff(cookies.pages, session.pages)



							var pageType = app.filters.type

							if(app.filters.type && session.type){
								if(app.filters.type != session.type){
									return
								}
							}


							if(window.footprint){
								if(window.footprint.href && session.href){
									if(window.footprint.href.toLowerCase() != session.href.toLowerCase()){
										return
									}
								}
							}

								


							app.block.fetch = false


							


							var pages = {}

							var detail = {}

							var primarys = []

							var foreigns = []

							var drafts = []

							var users = []

							var items = []

							var crons = []

							var origin_body = cookies.address ? `<option value="${app.host}">${app.host}</option>` : ''


							var drafts_body = ''

							var primarys_body = ''

							var foreigns_body = ''

							var pages_body = ''

							var users_body = ``

							var talks_body = ''


							var talks = []

							var $talks = $app.querySelectorAll(`[class*="${selector.talk}"]`)

							var talks_total = $talks.length

							var talks_index = $talks.length

							var talks_paginate = cookies.talks % 10

							console.log('isDetail',isDetail,app.filters.page);
							console.log('pageId',pageId);



							if(results?.length){
								var complete = {}

								// crons 값 동기화
								for(var i = 0; i < results.length; i++){
									var item = results[i]

									var bcc = hashId(item.type+(isDetail ? item.cc.toUpperCase() : item.cc))

									var ref = ''

									if(item.data){
										ref = hashId(item.to+item.cc+item.data.link)
									}

									if(item.table == "tasks"){

										if(item.data == null){
											if(app.filters.scan == item.id){
												delete app.filters.scan
											}

											console.log('cron.id',item.id);

											try{
												await Delete['crons']({
													key : 'id',
													value : item.id
												})
											}catch(err){
												console.log('crons err',err);
											}


											if($app.classList.contains(selector.scanning)){
												$app.classList.remove(selector.scanning)
											}

										}else{
											crons.push(item)

											console.log('item.data.contentType',item.data.contentType);

											try{
												await Upsert['crons'](item)
											}catch(err){

											}

											var task = item.data

											if(task.scan){
												item.data.text = `${(task.contentType.indexOf('image') > -1 ? 'image' : 'html')} Loading...`
											}else{
												item.data.text = `Prompt Loading...`
											}

											talks.push(item)
										}



										
											
									}


									if(item.table == "draft"){
										drafts.push(item)
									}

									if(item.table == "items"){
										// if(item.type == "prompt"){
										// 	if(item.data){
										// 		if(item.data.text){
										// 			var $bool = shadowRoot.querySelector(`[id="${item.id}"]`)

										// 			if(!$bool){
										// 				talks.push(item)
										// 			}
										// 		}

										// 	}
										// }

										if(item.data){
											if(item.data.text){
												var $bool = shadowRoot.querySelector(`[id="${item.id}"]`)

												if(!$bool){
													talks.push(item)
												}
											}

										}

										try{
											await Upsert['talks'](item);
										}catch(err){

										}
									}



									if(
										item.table == "sales" ||
										item.table == "event" ||
										item.table == "tracking"
									){
										try{
											await Upsert['items'](item);
										}catch(err){

										}

										if(item.status){
											if(item.status === 9 || item.status === 2 || item.status === 3){
												complete[item.id] = true
											}

											if(item.status === 10){
												drafts.push(item)
											}
										}

										// if(refererId){
										// 	if(pageId == refererId){
										// 		if(itemId == item.id){
										// 			detail = item
										// 		}else if(pageId == item.ref){
										// 			primarys.push(item)
										// 		}else{
										// 			foreigns.push(item)
										// 		}

										// 		continue
										// 	}
										// }


										if(itemId == item.ref && page.type == item.type){
											detail = item

										}else if(bcc == item.bcc || ref == item.ref){
											primarys.push(item)

										}else{
											foreigns.push(item)

										}
									}


									if(item.table == "talks"){
										// talks
										var $bool = shadowRoot.querySelector(`[id="${item.id}"]`)

										if(!$bool){
											talks.push(item)
										}
									}


									if(item.table == "pages"){
										try{
											if(item.data){
												pages[item.id] = item

												if(item.current){
													app.filters.page = page = item
													app.filters.type = page.type = item.type

													if(!app.filters.origin && item.data.origin){
														app.filters.origin = item.data.origin
													}

													for (const key in item.data) {
														if (item.data.hasOwnProperty(key)) {
															if(key == "text"){
																
															}else{
																try{
																	var $el = document.querySelector(item.data[key])

																	if(!$el){
																		item.data[key] = item.data[key].replace(/>/gi, " ")
																	}

																}catch(err){

																}

																try{
																	selector[key] = item.data[key]
																}catch(err){

																}
															}
														}
													}


													// if(selector.item){
													// 	var $talks = document.querySelectorAll(selector.item)

													// 	// console.log('talks.length',talks.length);

													// 	if($talks.length == 0){
													// 		if(refererId){
													// 			if(pageId == refererId){
													// 				throw new SyntaxError("continue"); 
													// 			}
													// 		}else{
													// 			throw new SyntaxError("continue");
													// 		}
													// 	}
													// }

												}
											}
										}catch(err){
											// console.log('page err',err);
										}

										try{
											if(!app.pages[item.id]){
												await Upsert['pages'](item);
											}
										}catch(err){
											console.log('err',err);
										}

									}


									if(item.table == "users"){
										users.push(item)

										item.data = {
											name : item.name
										}

										await Upsert['users'](item)
									}

								}

								var tasks = {}

								console.log('items',items);

								console.log('tasks',tasks);
								if(Object.keys(tasks).length){
									for (const task in tasks) {
										if (tasks.hasOwnProperty(task)) {
											var records = tasks[task]

											if(records.length){
												for(var r = 0; r < records.length; r++){
													var record = records[r]

													if(record.id == pageId || !app.pages[pageId]){
														var $bool = shadowRoot.querySelector(`[id="${record.id}"]`)

														if(!$bool){
															talks.push(record)
														}
													}
												}
											}	
										}
									}
								}

								var tmp = {}

								var inviteForm = ''

								async function renderAccordion(cookies, nodes, level = 1) {
									let html = `<ul class="${selector.branch}">`;

									for(var n = 0; n < nodes.length; n++){

										var node = nodes[n]

										var node_type = ''

										var active = ''

										var host = ''

										var type = ''

										var content = ''

										var name = ''

										var desc = []


										if(!tmp[node.id]){
											tmp[node.id] = true

											if(node.name){
												type = node.type

												name = node.name

												if(node.type == "team"){

													var teamName = node.name

													if(node.from == cookies.address && node.id == node.to){
														teamName = "Members"

														content = "Edit"
													}

													host = `<strong>${teamName}</strong>`

													inviteForm = `
														<label for="${selector.invite}" class="${selector.label}">+ Member</label>
														<div class="${selector.invite}">
															<input type="email" placeholder="E-Mail" required pattern="[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}$" />
															<input type="submit" onclick="window._${selector.invite}(event)" />
														</div>
													`
												}else{
													if(node.from == cookies.address && app.filters.team.from == cookies.address){
														desc.push("owner")
													}

													content = `<span>${name}${desc.length ? `<i>${desc.toString()}</i>` : ''}</span>`
												}

												if(node.id == app.filters.page.id){
													console.log('node.id, app.filters.page.id',node.id, app.filters.page.id);
													active = selector.active
												}

											}else if(node.data){

												type = 'page'

												name = `<span>${node.type}</span> <span>${(node.data.item ? " Draft" : " ")}</span>`

												if(node.data.origin){
													var _url = new URL(node.data.origin)

													if(!tmp[host] && node.data.item){
														host = `<strong>${_url.host}</strong>`
													}


													if(app.host.indexOf(_url.host) > -1){
														host += `<label for="${selector.membership}">Edit</label>`
													}


													if(app.filters.origin){
														
													}

													if(node.id == app.filters.page.id){
														active = selector.active

													}
													
												}


												// app.draft



												var total = {
													draft : 0,
													count : 0
												}

												console.log('node.cc',node.cc);

												console.log('cookies.pages',cookies.pages);

												if(cookies.pages){
													if(Object.keys(cookies.pages).length){
														if(cookies.pages[node.cc]){
															if(cookies.pages[node.cc][node.type]){
																total = cookies.pages[node.cc][node.type]
															}
														}
													}
												}

												console.log('total',total);


												var recent = ''

												try{
													var _items = await Select['items']({
														key : 'bcc',
														value : node.bcc
													})

													console.log('sssssss_items.length',_items.length);

													if(_items.length){
														var _item = _items[0]

														var _user = null

														if(app.users[_item.from]){
															_user = app.users[_item.from]
														}else{
															var _users = await Select['users']({
																key : 'id',
																value : _item.from
															})

															if(_users.length){
																_users = _users[0]
															}
														}

														if(_user){
															recent = `<strong>${time2text(_item.data.time ? _item.data.time : _item.created_at)} - ${_user.name}</strong>`
														}
														
													}
												}catch(err){

												}




												var count = ''

												if(node.data.item){
													count = `<u>(${total.draft})</u>`
												}else{
													count = `<u>(${total.count})</u>`
												}

												content = `<span>${name}
													${count}
												</span>
												${recent}
												`
											}
										}

										var hasChildren = node.children.length > 0;


										html += `<input type="checkbox" name="${type}" id="${type}-${node.id}" />
										<li class="${selector.parent} ${hasChildren ? selector.children : ''}" ${type}-id="${node.id}">`;
										
										// header
										html += `
											${host}
											<label for="${type}-${node.id}" class="${selector.label} ${type}-${node.id} ${active}">
												${content}
											</label>
										`;

										if(hasChildren){
											// body
											html += `<div class="${selector.child} ${type}-${node.id}">`;

											if(node.type == "team"){
												html += `<form name="${selector.team}">
													<input type="text" value="${node.name}" required>
													<input type="submit">
												</form>`
											}


											if (hasChildren) {
												html += await renderAccordion(cookies, node.children, level + 1);
											}

											html += `</div>`;
										}

											
										html += `</li>`;
									}

									html += `</ul>`;

									return html;
								}


								console.log('users',users);

								if(users.length){
									var temp = {};

									var tree = [];

									for (var u = 0; u < users.length; u++) {
										var user = users[u]

										if(!app.users[user.id]){
											userDiff = true
										}else{
											userDiff = isDiff(app.users[user.id], user)
										}
										
										app.users[user.id] = user

										if(pageId == user.id){
											app.filters.page = user
										}


										if(user.type == "team" && cookies.team == user.id){
											app.filters.team = user
										}

										temp[user.id] = { ...user, children: [] }
									}


									if(Object.keys(app.users).length != users.length){
										userDiff = true
									}


									if(userDiff){
										for(var key in temp){
											if (temp.hasOwnProperty(key)) {
												var user = temp[key]

												var parentId = user.to

												if(user.type == "user"){
													temp[parentId].children.push(temp[key])

												}else if(user.type == "team"){
													tree.push(temp[key])

												}
											}
										}

										if(Object.keys(tree).length){
											users_body += await renderAccordion(cookies, tree)
											users_body += inviteForm
										}
									}
								}



								if(talks.length){
									console.log('talks',talks);

									delete app.block.talks

									talks = talks.sort((a, b) => a.created_at - b.created_at);

									for(var t = 0; t < talks.length; t++){
										var talk = talks[t]

										// console.log('talk',talk);

										var id = talk.id ? talk.id : selector.none

										// if(talk.type == "prompt" && talk.data == null){
										// 	continue
										// }

										var status = ""

										var created_at = new Date(talk.created_at - timezoneOffset).toISOString()
											created_at = created_at.split("T")
											created_at = created_at[0]


										var seed = talk.from

										if(!seed){
											seed = cookies.address
										}

										var name = ''

											

										console.log('_users_users_users',_users,talk.from);

										var _user = null

										if(app.users[talk.from]){
											_user = app.users[talk.from]
										}else{
											var _users = await Select['users']({
												key : 'id',
												value : talk.from
											})

											if(_users.length){
												_users = _users[0]
											}
										}

										if(_user){
											console.log('_user_user_user_user',_user);
											name = ' / @'+_user.name
										}


										var canvas = blockies.create({seed: seed.toLowerCase()})
										
										var base64 = canvas.toDataURL()

										var isMarkdown = talk.data.markdown ? true : false


										talks_body += `<input type="checkbox" class="${selector.checkbox}" id="${talk.id.toUpperCase()}" />`

										// console.log('talk.type',talk.type);

										talks_body += `<div id="${talk.id}" data-ref="${talk.ref}" class="${selector.talk} ${isMarkdown ? selector.markdown : ''} ${talk.type == "talk" ? selector.user : selector.system} ${(selector["type_"+talk.type] ? selector.system : "")}">
											<label for="${talk.id.toUpperCase()}" class="${selector.label}"><span>${status} ${talk.type} ${name}</span></label>
											<div class="${selector.content}">`

										if(talk.data){
											if(talk.data.markdown){
												talks_body += `<div class="${selector.message}">
													<deco style="background-image: url(${base64});"></deco>
													<text>${marked(talk.data.markdown)}</text>
												</div>`
											}else if(talk.data.text){
												talks_body += `<div class="${selector.message}">
													<deco style="background-image: url(${base64});"></deco>
													<text>${talk.data.text}</text>
												</div>`
											}
										}
									

										var paginate = ''

										talks_index = talks_index + 1

										if(
											(talks_index % 10 == 0) || 
											(talks_total == 0 && (i == talks_paginate))
										){
											paginate = `<div class="${selector.paginate}">
												<strong>${talks_index}</strong> / <span>${cookies.talks}</span>
											</div>`
										}

										talks_body += `</div>
											<span class="${selector.created_at}">${created_at}</span>
											<input type="hidden" readonly name="${selector.created_at}" value="${talk.created_at}" />
											${paginate}
										</div>`
									}
										
								}else if(event == 'talks'){
									app.block.talks = true
								}

								
								

								var $target = $app.querySelector(`[class*="${selector.results}"] [class*="${selector.scroll}"]`)

								console.log('pageDiff',pageDiff, !$target.innerHTML);

								if(pageDiff || !$target.innerHTML){
									console.log('detail',detail);

									if(Object.keys(detail).length){
										var $item = shadowRoot.querySelector(`[id="more-${detail.id}"]`)

										if(!$item){
											$target.innerHTML += item2html(detail, true)
										}
										
									}


									if(primarys.length == 0 && foreigns.length == 0){
										app.block.items = true
									}else{
										delete app.block.items									}

									if(primarys.length){
										for(var i = 0; i < primarys.length; i++){
											var primary = primarys[i];

											if(!app.items[primary.ref]){
												app.items[primary.ref] = primary
											}

											primarys_body += item2html(primary, null, (Object.keys(detail).length ? (primary.id == detail.id || primary.ref == detail.ref) : false))

										}
									}
									

									if(primarys_body){
										$target.innerHTML += primarys_body
									}

									if(foreigns.length){
										for(var i = 0; i < foreigns.length; i++){
											var foreign = foreigns[i];

											if(!app.items[foreign.ref]){
												app.items[foreign.ref] = foreign
											}

											var value = foreign[app.filters.type]

											if(value){
												var $relate = $target.querySelector(`[${foreign.type}="${value}"]`)

												if($relate){
													$relate.innerHTML += item2html(foreign, true, true)
												}
											}
										}
									}

									if(drafts.length){
										for(var i = 0; i < drafts.length; i++){
											var draft = drafts[i];

											var value = drafts[app.filters.type]

											if(value){
												var $relate = $target.querySelector(`[${draft.type}="${value}"]`)

												if($relate){
													$relate.innerHTML += item2html(draft)
												}else{
													// 없으면 맨위에 새로 추가해야함

													$target.prepend(item2html(draft))
												}
											}
										}
									}
								}



								var talks = shadowRoot.querySelector(`[class*="${selector.talks}"]`)

								if(talks_body.length){
									talks.innerHTML += talks_body
								}

								


								

								if(pageDiff || Object.keys(app.pages).length == 0){
									var _pages = await Select['pages']()

									var _len = _pages.length

									console.log('_pages',_pages);

									var now = Date.now()


									if(_pages.length){
										var branchs = []

										var options = {}


										for (var p = 0; p < _pages.length; p++) {
											var _page = _pages[p]

											var _url = new URL(_page.data.origin)

											if(!options[_url.host]){
												options[_url.host] = true

												origin_body += `<option ${app.filters.origin == _page.data.origin ? 'selected' : ''} value="${_url.origin}">${_url.host}</option>`
											}

											if(!Ethers.isAddress(_page.id)){
												continue
											}

											if(app.pages[_page.id]){
												continue
											}

											app.pages[_page.id] = _page

											if(pageId == _page.id){
												app.filters.page = _page
											}

											if(_page.data.item){
												branchs[`${_page.data.origin}#${_page.type}`] = { ..._page, children: [] }
											}

											branchs[_page.id] = { ..._page, children: [] }
										}


										var temp = {}

										for(var key in branchs){
											if (branchs.hasOwnProperty(key)) {
												var _page = safeClone(branchs[key])

												if(!temp[_page.id]){
													temp[_page.id] = true

													var parent = branchs[`${_page.data.origin}#${_page.type}`]

													if(parent){
														if(_page.data.item){
															var children = safeClone(parent.children)

															branchs[`${_page.data.origin}#${_page.type}`] = {
																..._page,
																children : children
															}
														}else if(!temp[`${_page.data.origin}#${_page.type}`]){
															temp[`${_page.data.origin}#${_page.type}`] = true

															branchs[`${_page.data.origin}#${_page.type}`].children.push(_page)
														}else if(typeof _page.data.node == "string" && typeof _page.data.item == "string" && !_page.data.item && typeof temp[`${_page.data.origin}#${_page.type}`] != "string"){

															if(branchs[`${_page.data.origin}#${_page.type}`].children.length){
																temp[`${_page.data.origin}#${_page.type}`] = 'true'

																var index = 0

																for(var b = 0; i < branchs[`${_page.data.origin}#${_page.type}`].children.length; b++){
																	var _item = branchs[`${_page.data.origin}#${_page.type}`].children[b]

																	if(_page.data.node === true){
																		index = b
																	}
																}

																branchs[`${_page.data.origin}#${_page.type}`].children.splice(index, 1);

																branchs[`${_page.data.origin}#${_page.type}`].children.push(_page)
															}
														}
														
													}else{
														if(_page.data.item){
															if(!branchs[`${_page.data.origin}#${_page.type}`]){
																branchs[`${_page.data.origin}#${_page.type}`] = {
																	..._page,
																	children : []
																}
															}
														}else if(!temp[`${_page.data.origin}#${_page.type}`]){
															temp[`${_page.data.origin}#${_page.type}`] = true
															console.log('detail page진입',_page);
															branchs[`${_page.data.origin}#${_page.type}`].children.push(_page)
														}else if(typeof _page.data.node == "string" && typeof _page.data.item == "string" && !_page.data.item && typeof temp[`${_page.data.origin}#${_page.type}`] != "string"){

															if(branchs[`${_page.data.origin}#${_page.type}`].children.length){
																temp[`${_page.data.origin}#${_page.type}`] = 'true'

																var index = 0

																for(var b = 0; i < branchs[`${_page.data.origin}#${_page.type}`].children.length; b++){
																	var _item = branchs[`${_page.data.origin}#${_page.type}`].children[b]

																	if(_page.data.node === true){
																		index = b
																	}
																}

																branchs[`${_page.data.origin}#${_page.type}`].children.splice(index, 1);

																branchs[`${_page.data.origin}#${_page.type}`].children.push(_page)
															}
														}
													}


															
												}
											}
										}

										console.log('branchs',branchs);

										var tree = []

										for(var key in branchs){
											if (branchs.hasOwnProperty(key)) {
												if(!Ethers.isAddress(key)){
													var _page = branchs[key]

													tree.push(_page)
												}
											}
										}

										console.log('tree',tree);

										if(tree.length){
											pages_body = await renderAccordion(cookies, tree)
										}
									}
								}



								if(pages_body){
									var $pages = shadowRoot.querySelector(`[class*="${selector.pages}"]`)

									$pages.innerHTML = pages_body

									if(origin_body){
										var $origin = shadowRoot.querySelector(`[name="${selector.host}"]`)

										console.log('$origin',$origin,`[name="${selector.host}"]`);

										if($origin){
											$origin.innerHTML = origin_body
										}
									}
								}

								if(userDiff){
									var $users = shadowRoot.querySelector(`[class*="${selector.users}"]`)

									if($users.innerHTML != users_body){
										$users.innerHTML = users_body
									}
								}



								if(!pageType && app.filters.type){
									$app.setAttribute(selector.page, selector['type_'+app.filters.type])

									$app.querySelector(`[id="${selector.page}"]`).textContent = app.filters.type

								}


								console.log('selector.item',selector.item);

								if(selector.item){
									var $items = document.querySelectorAll(selector.item)

									

									var list = []

									var $links = []

									console.log('$items.length',$items.length);

									if($items.length){
										for(var i = 0; i < $items.length; i++){
											var $item = $items[i]

											var $list = $item.querySelectorAll('[href]')

											if($list.length){
												for(var s = 0; s < $list.length; s++){
													var $link = $list[s]

													$link.setAttribute('target', "_blank")

													$link.removeAttribute('rel')
													$link.removeAttribute('referrerpolicy')

													$link.href = $link.href.toLowerCase()

													$links.push($link)

													// var $footprint = document.createElement('div');
												}
											}

											if(selector.more){
												// console.log('$item',$item, selector.more);
												var $mores = $item.querySelectorAll(selector.more)

												if($mores.length){
													if($mores.length){
														for(var m = 0; m < $mores.length; m++){
															var $more = $mores[m]

															try{
																var more = new URL($more.href)


																$more.setAttribute('href', more.href)

																list.push(more.href)

																break;
															}catch(err){
																// console.log('err',err);
															}
														}
													}
												}
											}


											//  query[]

											/*
												style 수정하고

												안내 내용 마크업 추가해야함
											*/ 
										}

										console.log('selector',selector);

										console.log('selector.node, list.length',selector.node, list);

										console.log('selector.node',selector.node);

										if(!pageType && app.filters.type){
											pageType = app.filters.type
										}

										if(selector.node && list.length){
											var created_at = new Date(new Date().getTime() - timezoneOffset).getTime()

											var params = ``

											if(document.referrer){
												try{
													var referrer = new URL(document.referrer)

													if(window.location.href.indexOf(referrer) > -1){
														params = `&referrer=${encodeURIComponent(document.referrer)}`
													}
												}catch(err){

												}
											}

											var { results, session } = await app.fetch({
												url : reqUrl( cookies, app.filters, { href : window.location.href } ),
												method: "POST",
												headers: {
													'Content-Type': 'application/octet-stream',
													'Content-Encoding': 'gzip'
												},
												body : JSON.stringify(list)
											})
											

											if(session){
												cookies = session
											}

											var sync = []

											var isNew = false


											var temp = {}

											temp[pageId] = footprint.href

											for(var a = 0; a < $links.length; a++){
												var $link = $links[a]

												try{
													var _url = new URL($link.href)

													var _id = hashId(cookies.team+cookies.cc+_url.pathname+_url.search)

													temp[_id] = $link.href

												}catch(err){
													
												}
											}

											console.log('temp',temp);

											console.log('results',results);

											if(results.length){
												for(var r = 0; r < results.length; r++){
													var item = results[r]

													var visited = false

													
													if(item.data){
														if(item.data.link){
															var _url = new URL(item.data.origin+item.data.link)

															var ref = hashId(cookies.team+cookies.cc+_url.pathname+_url.search)

															var bcc = hashId(item.type+cookies.cc.toUpperCase())

															var _items = await Select['items']({
																key : 'id',
																value : item.id
															})

															

															if(_items.length){
																var _item = _items[0]

																console.log('_item.type,pageType',_item.type,pageType, _item.bcc, bcc);

																// if(_item.ref == pageId){
																if(_item.type == pageType){
																	if(bcc == _item.bcc){
																		visited = item.data.origin+item.data.link
																	}
																}
															}
														}
													}

													var active = temp[item.id]

													var completed = complete[item.id]

													var scan = visited || active

													console.log('scan,visited,active',scan,visited,active);

													if(scan){
														sync.push(scan)
													}else{
														isNew = true
													}


													for(var s = 0; s < $items.length; s++){
														var $this = $items[s]

														var $link = $this.querySelector(`[href*="${item.data.link}"]`)

														if($link){
															var $this = $link.closest(selector.item)

															var rowspan = 0

															var $rowspan = $link.closest(`[rowspan]`)

															if(!$rowspan){
																$rowspan = $this.querySelector('[rowspan]')

																if($rowspan){
																	rowspan = $rowspan.getAttribute('rowspan') * 1
																}
															}

															

															if(visited){
																if(completed){
																	console.log('성공 진입 completed');
																	$this.classList.add(selector.completed)

																	if(rowspan > 1 && $rowspan && $this.nextElementSibling){
																		$this.nextElementSibling.classList.add(selector.completed)
																	}
																}else{
																	$this.classList.add(selector.visited)

																	if(rowspan > 1 && $rowspan && $this.nextElementSibling){
																		$this.nextElementSibling.classList.add(selector.visited)
																	}
																}

															}else if(active){
																$this.classList.add(selector.active)

																if(rowspan > 1 && $rowspan && $this.nextElementSibling){
																	$this.nextElementSibling.classList.add(selector.active)
																}

															}else{
																console.log('예외',$this);
															}
														}else{
															isNew = true
														}
													}
												}
											}

											// console.log('sync',sync);
											// console.log('list',list);

											// 클릭 안된 링크
											var hiddens = list.filter(link => !sync.includes(link));

											// console.log('hiddens',hiddens);

											if(hiddens.length){
												for(var i = 0; i < hiddens.length; i++){
													try{
														var url = new URL(hiddens[i])
														var hidden = url.pathname + url.search

														var $link = document.querySelector(`[href*="${hidden}"]`)

														if($link){
															var $this = $link.closest(selector.item)

															var $rowspan = $link.closest(`[rowspan]`)

															if(!$rowspan){
																$rowspan = $this.querySelector('[rowspan]')
															}

															$this.classList.add(selector.$item)

															if($rowspan && $this.nextElementSibling){
																$this.nextElementSibling.classList.add(selector.$item)
															}
														}

													}catch(err){
														console.log('err',err);
													} 
												}
											}
										}
									}
								}
							}

							console.log('foreigns',foreigns)

							console.log('primarys',primarys)
						}
					}

					if(isFocus && cookies.address || retryCount){
						window[cookies.hash] = setTimeout(timeout.fn, timeout.ms)
					}
				}catch(err){
					console.log('err',err);
				}
			}
		}

		console.log(isClient, isAdmin, app.chrome)

		if(isClient || isAdmin || !app.chrome){
			var $input = document.createElement('input')

			$input.type = "checkbox"
			$input.id = selector.left
			
			$app.before($input)


			var seed = cookies.address ? cookies.address : Ethers.ZeroAddress;

			var canvas = blockies.create({seed: seed.toLowerCase()});
			
			var base64 = canvas.toDataURL();

			var option_body = ''

			if(window.location.host && cookies.address){
				option_body = `<option selected value="${window.location.host}">${window.location.host}</option>`
			}



			$app.innerHTML = `
				<div class="${selector.dim}"></div><input type="checkbox" id="${selector.right}" />
				<div class="${selector.area}">
					<a id="${selector.scan}">
						<i></i>
					</a>
					<input type="checkbox" id="${selector.header}">
					<div class="${selector.header}">
						<div class="${selector.host}">
							<select name="${selector.host}" id="${selector.host}" disabled>${option_body}</select>
						</div>
						<div id="${selector.page}"></div>
						<div id="${selector.interval}">
							<select name="${selector.interval}">
								<option selected value="">Recent</option>
								<option value="week">Week</option>
								<option value="month">Month</option>
							</select>
						</div>
					</div>
					<div class="${selector.aside} ${selector.left}">
						<input type="checkbox" id="${selector.membership}">
						<div id="${selector.filters}">
							<input id="${selector.invite}" type="checkbox">
							<div class="${selector.filters}">
								<label for="${selector.pip}"></label>

								<input type="checkbox" id="${selector.setting}">
								<div class="${selector.profile}">
									<div class="${selector.info}">
										<div class="${selector.favicon}" style="background-image: url(${base64});"></div>
										<div class="${selector.name}">
											<form name="${selector.setting}">
												<input type="text" value="${cookies.name}" required>
												<input type="submit">
											</form>
											<strong>${cookies.name ? cookies.name : 'Sign In'}</strong>
											<label for="${selector.setting}">Edit</label>
										</div>
									</div>

									<a class="${selector.menu}" id="${selector.sign.in}">Sign In</a>
									<a class="${selector.menu}" id="${selector.sign.out}">Sign Out</a>
								</div>

								<div class="${selector.scroll}">
									<div class="${selector.membership}"></div>

									<div class="${selector.pages}"></div>

									<div class="${selector.users}"></div>

				
								</div>
							</div>
						</div>
					</div>
					<div class="${selector.center}">
						<input type="checkbox" id="${selector.address}">

						<div class="${selector.results}">
							<label for="${selector.pip}"></label>

							<div class="${selector.prompt}">								
								<form name="${selector.prompt}">

									<textarea maxlength="140" cols="40" rows="2" name="${selector.context}" placeholder="${placeholder.prompt()}"></textarea>

									<label for="${selector.file}">Image Upload</label>

									<input type="file" id="${selector.file}" accept="image/*">
									<input type="reset" id="${selector.reset}" value="reset">

									<label for="${selector.submit}">
										${logo}
										<input id="${selector.submit}" type="submit">
									</label>
								</form>

								<div class="${selector.sender}">
									<label for="${selector.address}">Setting Sender</label>
								</div>
							</div>

							<form name="${selector.address}">
								<textarea name="${selector.sender}" required placeholder="Sender Address List"></textarea>
								<input type="submit">
							</form>

							<div class="${selector.scroll}"></div>
						</div>

						<div class="${selector.scan} ${selector.ocr}">
							<video id="${selector.video}" width="460" height="818" autoplay playsinline></video>
							<canvas id="${selector.canvas}"></canvas>
							
							<img id="${selector.photo}" alt="camera photo" style="display: none;">

							<div class="buttons">
								<button id="${selector.parse}">Snap</button>
							</div>
							
							<div id="${selector.camera}">Camera Open</div>
						</div>
					</div>

					<div class="${selector.aside} ${selector.right}">
						<div class="${selector.qrcode}"></div>
						<a class="${selector.qrauth}">QR Verify</a>
						
						<input type="checkbox" id="${selector.system}" checked>
						<div class="${selector.chat}">
							<label for="${selector.system}">System Message<i class="${selector.count}"></i></label>
							
							<div class="${selector.scroll}">
								<div class="${selector.talks}"></div>
								<div class="${selector.bottom}"></div>
							</div>
							<form name="${selector.chat}">
								<input type="text" name="${selector.talk}" placeholder="chat message" required>
								<input type="submit">
							</form>
						</div>
					</div>
					<label for="${selector.right}"></label>
				</div>
				<label for="${selector.left}" status="off"></label>
			`



			/*
				<div class="${selector.teams}">
					<div class="${selector.active}"></div>
					<div class="${selector.idle}"></div>
				</div>
			*/ 	

		
			var $left = $app.querySelector(`[for="${selector.left}"]`)
			$left.addEventListener('click', async function(e){
				var { cookies } = await app.storage.get('cookies')

				if(app.chrome){
					e.preventDefault()
			
					var pageId = hashId(cookies.team+cookies.cc+window.location.pathname)

					var scanId = hashId(cookies.team+cookies.cc+window.location.pathname+window.location.search)

					var crons = await Select['crons']({
						key : 'ref',
						value : scanId
					})

					console.log('crons',crons);

					console.log('app.filters.scan',app.filters.scan);

					if(crons.length && app.filters.scan){
						var bool = window.confirm('Scan Cancel?')

						if(bool){
							await Delete['crons']({
								key : 'id',
								value : app.filters.scan
							})

							if($app.classList.contains(selector.scanning)){
								$app.classList.remove(selector.scanning)
							}

							var { results } = await app.fetch({
								url : reqUrl( cookies, app.filters, { type:'crons', ref: pageId } ),
								method: 'DELETE'
							})
						}

					}else{
						try{
							$app.classList.add(selector.scanning)

							setStates()

							var $el = document.createElement('div')
								$el.innerHTML = document.body.innerHTML
								$el.querySelector(`[class*="${selector.app}"]`).remove()

							var $garbeges = [...$el.querySelectorAll('script'), ...$el.querySelectorAll('style'), ...$el.querySelectorAll('link'), ...$el.querySelectorAll('noscript'), ...$el.querySelectorAll('iframe')]

							if($garbeges.length){
								for(var g = 0; g < $garbeges.length; g++){
									$garbeges[g].remove()
								}
							}

							var body = $el.innerHTML.replace(/<!--[\s\S]*?-->/g, '').trim();
							
							cleanStates()

							
							app.filters.scan = hashId(cookies.team+cookies.cc+body)



							
							console.log('{ from: cookies.address, to : cookies.team }',{ from: cookies.address, to : cookies.team });

							var { results } = await app.fetch({
								url : reqUrl( cookies, app.filters, { from: cookies.address, to : cookies.team, href : window.location.href } ),
								method: 'POST',
								headers: {
									'Content-Type': 'application/octet-stream',
									'Content-Encoding': 'gzip'
								},
								body : body
							})

							console.log('app.filters.scan',app.filters.scan);

							if(results.success){
								try{
									await Upsert['crons']({
										id : app.filters.scan,
										cc : scanId,
										bcc : scanId,
										ref : scanId,
										job : null,
										created_at : 0,
										updated_at : 0
									})
								}catch(err){

								}
							}

								

							console.log('results',results)

							// $app.classList.remove(selector.loading)
						}catch(err){
							console.log('err',err);
						}
					}
					
				}else{
					console.log('right click')

					if(cookies.address){
						shadowRoot.querySelector(`[id="${selector.right}"]`).checked = false
					}else{
						e.preventDefault()

						shadowRoot.querySelector(`[id="${selector.left}"]`).checked = false
						shadowRoot.querySelector(`[id="${selector.right}"]`).checked = true
					}

				}
					
			})


			

			var $reset = $app.querySelector(`[id="${selector.reset}"]`)


			$reset.addEventListener('click', function (e) {
				var $prompt = $app.querySelector(`form[name="${selector.prompt}"]`)

				app.upload = {}

				$prompt.removeAttribute('style')
			})

			var $file = $app.querySelector(`[id="${selector.file}"]`)
			


			$file.addEventListener('change', async function (e) {
				var $this = e.target
				var file = $this.files[0];

				var { cookies } = await app.storage.get('cookies')

				if(!cookies.sender){
					$app.querySelector(`[id="${selector.reset}"]`).click()
					$app.querySelector(`[for="${selector.address}"]`).click()

					return
				}

				console.log('e.target',e.target);

				var $prompt = $app.querySelector(`form[name="${selector.prompt}"]`)

				var $context = $prompt.querySelector(`[name="${selector.context}"]`)
					$context.value = ""

				// FileReader 객체 생성
				const reader = new FileReader();

				// 파일 읽기 완료 시 이벤트 핸들러
				reader.onload = function(e) {
					var base64String = e.target.result; 

					console.log('e.target',e.target);

					app.upload = {
						format : file.type,
						body : base64String
					}

					var $prompt = $app.querySelector(`form[name="${selector.prompt}"]`)

					// console.log('$prompt',$prompt,base64String);
					// $prompt.setAttribute('style', `background-image`)

					$prompt.style['background-image'] = `url(${base64String})`;
				};

				// **핵심: 파일을 Base64 데이터 URL로 읽도록 지시**
				reader.readAsDataURL(file);
			});

			


			var $scan = $app.querySelector(`[id="${selector.scan}"]`)

			$scan.addEventListener('click', async function(e){
				if(app.chrome){
					e.preventDefault()

					var { cookies } = await app.storage.get('cookies')


					if(!cookies.sender){
						$app.querySelector(`[for="${selector.address}"]`).click()

						return
					}
				

					var pageId = hashId(cookies.team+cookies.cc+window.location.pathname)

					var scanId = hashId(cookies.team+cookies.cc+window.location.pathname+window.location.search)

					var crons = await Select['crons']({
						key : 'ref',
						value : scanId
					})


					console.log('crons',crons);

					console.log('app.filters.scan',app.filters.scan);

					if(crons.length){
						var scan = crons[0]

						var bool = window.confirm('Cancel?')

						if(bool){
							if($app.classList.contains(selector.scanning)){
								$app.classList.remove(selector.scanning)
							}

							var { results } = await app.fetch({
								url : reqUrl( cookies, app.filters, { type:'crons', ref: pageId, href : window.location.href } ),
								method: 'DELETE'
							})

							await Delete['crons']({
								key : 'id',
								value : scan.id
							})
						}

					}else{
						try{
							setStates()

							var $el = document.createElement('div')
								$el.innerHTML = document.body.innerHTML
								
							var $garbeges = [...$el.querySelectorAll('script'), ...$el.querySelectorAll('style'), ...$el.querySelectorAll('link'), ...$el.querySelectorAll('noscript'), ...$el.querySelectorAll('iframe')]

							if($garbeges.length){
								for(var g = 0; g < $garbeges.length; g++){
									$garbeges[g].remove()
								}
							}

							var body = $el.innerHTML.replace(/<!--[\s\S]*?-->/g, '').trim();
							
							cleanStates()

							
							app.filters.scan = hashId(cookies.team+cookies.cc+body)

							console.log('app.filters.scan',app.filters.scan);


							var url = new URL(window.location.href)

							var filters = {}


							if(Object.keys(app.filters.page).length){
								var _link = url.pathname + url.search
								if(
									url.origin == app.filters.page.data.origin && 
									_link == app.filters.page.data.link
									){
									filters = app.filters
								}
								
							}
							
							console.log('{ from: cookies.address, to : cookies.team }',{ from: cookies.address, to : cookies.team });


							$app.classList.add(selector.scanning)

							$app.classList.add(selector.loading)

							// console.log('body',body);
							// return

							var { results } = await app.fetch({
								url : reqUrl( cookies, filters, { from: cookies.address, to : cookies.team, href : url.href, format : 'text/html' } ),
								method: 'POST',
								headers: {
									'Content-Type': 'application/octet-stream',
									'Content-Encoding': 'gzip'
								},
								body : body
							})

							console.log('results',results);

							try{
								await Upsert['crons']({
									id : app.filters.scan,
									cc : scanId,
									bcc : scanId,
									ref : scanId,
									job : null,
									created_at : 0,
									updated_at : 0
								})
							}catch(err){

							}
						}catch(err){
							console.log('err',err);
						}
					}
				}else{
					if(!app.chrome){
						shadowRoot.querySelector(`[id="${selector.left}"]`).checked = false
						shadowRoot.querySelector(`[id="${selector.right}"]`).checked = false
					}

					$app.classList.toggle(selector.scan)

					if($app.className.indexOf(selector.scan) > -1){
						var cametaStatus = $app.querySelector(`[id="${selector.camera}"]`);
						
						var video = $app.querySelector(`[id="${selector.video}"]`);

						var canvas = $app.querySelector(`[id="${selector.canvas}"]`);

						canvas.width = 480 // window.innerWidth
						canvas.height = 640 // window.innerHeight



						try {
							app.stream = await navigator.mediaDevices.getUserMedia({ 
								video: { 
									facingMode: 'environment',
									aspectRatio: { ideal: 1.3333 },
									height: { min: 480 }
								}, 
								audio: false 
							});
							video.srcObject = app.stream;
							video.play();
							cametaStatus.textContent = "카메라 준비 완료. 사진을 찍어주세요.";
						} catch (err) {
							console.error("카메라 접근 오류:", err);
							cametaStatus.textContent = "오류: 카메라에 접근할 수 없습니다. 권한을 확인해주세요.";
						}
					}else if(app.stream){
						app.stream.getTracks().forEach(track => track.stop());

						app.stream = null
					}
				}

			})





			var $interval = $app.querySelector(`[name="${selector.interval}"]`)

			$interval.addEventListener('change', async function(e){
				var $this = e.target


				var $target = $app.querySelector(`[class*="${selector.results}"] [class*="${selector.scroll}"]`)

				$target.innerHTML = ""

				var talks = shadowRoot.querySelector(`[class*="${selector.talks}"]`)

				talks.innerHTML = ''



				app.pages = {}

				var now = Date.now();

				if($this.value == "week"){
					app.filters.created_at = now - (7 * 24 * 60 * 60 * 1000);
				}else if($this.value == "month"){
					app.filters.created_at = now - (30 * 24 * 60 * 60 * 1000);
				}else{
					app.filters.created_at = now
				}


				$app.classList.add(selector.loading)

				timeout.clear()

				await timeout.fn()

				$app.classList.remove(selector.loading)
			})

			var $filters = $app.querySelector(`[id="${selector.filters}"]`)

			$filters.addEventListener('click', async function(e){
				if(app.block.fetch){
					e.preventDefault()
					return
				}

				var $this = e.target

				var id = $this.getAttribute('for')

				if(id == selector.membership){
						// start xag
						var _pages = await Select['pages']()

						var _len = _pages.length

						console.log('_pages',_pages);

						var now = Date.now()


						if(_pages.length){
							var branchs = []

							var options = {}


							for (var p = 0; p < _pages.length; p++) {
								var _page = _pages[p]

								var _url = new URL(_page.data.origin)

								if(!options[_url.host]){
									options[_url.host] = true
								}

								if(!Ethers.isAddress(_page.id)){
									continue
								}

								if(app.pages[_page.id]){
									continue
								}

								app.pages[_page.id] = _page

								if(pageId == _page.id){
									app.filters.page = _page
								}

								if(_page.data.item){
									branchs[`${_page.data.origin}#${_page.type}`] = { ..._page, children: [] }
								}

								branchs[_page.id] = { ..._page, children: [] }
							}


							var temp = {}

							for(var key in branchs){
								if (branchs.hasOwnProperty(key)) {
									var _page = safeClone(branchs[key])

									if(!temp[_page.id]){
										temp[_page.id] = true

										var parent = branchs[`${_page.data.origin}#${_page.type}`]

										if(parent){
											if(_page.data.item){
												var children = safeClone(parent.children)

												branchs[`${_page.data.origin}#${_page.type}`] = {
													..._page,
													children : children
												}
											}else if(!temp[`${_page.data.origin}#${_page.type}`]){
												temp[`${_page.data.origin}#${_page.type}`] = true

												branchs[`${_page.data.origin}#${_page.type}`].children.push(_page)
											}else if(typeof _page.data.node == "string" && typeof _page.data.item == "string" && !_page.data.item && typeof temp[`${_page.data.origin}#${_page.type}`] != "string"){

												if(branchs[`${_page.data.origin}#${_page.type}`].children.length){
													temp[`${_page.data.origin}#${_page.type}`] = 'true'

													var index = 0

													for(var b = 0; i < branchs[`${_page.data.origin}#${_page.type}`].children.length; b++){
														var _item = branchs[`${_page.data.origin}#${_page.type}`].children[b]

														if(_page.data.node === true){
															index = b
														}
													}

													branchs[`${_page.data.origin}#${_page.type}`].children.splice(index, 1);

													branchs[`${_page.data.origin}#${_page.type}`].children.push(_page)
												}
											}
											
										}else{
											if(_page.data.item){
												if(!branchs[`${_page.data.origin}#${_page.type}`]){
													branchs[`${_page.data.origin}#${_page.type}`] = {
														..._page,
														children : []
													}
												}
											}else if(!temp[`${_page.data.origin}#${_page.type}`]){
												temp[`${_page.data.origin}#${_page.type}`] = true
												console.log('detail page진입',_page);
												branchs[`${_page.data.origin}#${_page.type}`].children.push(_page)
											}else if(typeof _page.data.node == "string" && typeof _page.data.item == "string" && !_page.data.item && typeof temp[`${_page.data.origin}#${_page.type}`] != "string"){

												if(branchs[`${_page.data.origin}#${_page.type}`].children.length){
													temp[`${_page.data.origin}#${_page.type}`] = 'true'

													var index = 0

													for(var b = 0; i < branchs[`${_page.data.origin}#${_page.type}`].children.length; b++){
														var _item = branchs[`${_page.data.origin}#${_page.type}`].children[b]

														if(_page.data.node === true){
															index = b
														}
													}

													branchs[`${_page.data.origin}#${_page.type}`].children.splice(index, 1);

													branchs[`${_page.data.origin}#${_page.type}`].children.push(_page)
												}
											}
										}


												
									}
								}
							}

							console.log('branchs',branchs);

							var tree = []

							for(var key in branchs){
								if (branchs.hasOwnProperty(key)) {
									if(!Ethers.isAddress(key)){
										var _page = branchs[key]

										tree.push(_page)
									}
								}
							}

							if(tree.length){

								var container = shadowRoot.querySelector(`[class*="${selector.membership}"]`);

								// Clear existing content if any
								container.innerHTML = '';

								// Loop through the array and create a new element for each tag
								for(var t = 0; t < tree.length; t++){
									var page = tree[p]

									var $tag = document.createElement("span");
									
									// Set the text content of the element
									$tag.textContent = tagName;
									
									// Add some basic styling (you can also use CSS classes)
									$tag.className = selector.aaaa

									// Optional: Add an event listener for interaction
									$tag.addEventListener("click", () => {
										alert(`You clicked on the tag: ${tagName}`);
									});

									// Append the new element to the container in the HTML body
									container.appendChild($tag);
								}
							}
						}

						// end xag


				}else if($this.classList.contains(selector.label)){
					if(id.indexOf('page-') > -1){

						id = id.replace('page-',"")

						console.log('app.filters.page',app.filters.page);

						app.block.fetch = true

						var page = app.pages[id]

						console.log('page',page);

						if(!page){
							return
						}


						app.filters.page = page

						app.filters.type = page.type

						app.filters.origin = page.data.origin

						app.filters.created_at = Date.now()

						var $option = $app.querySelector(`select[name="${selector.host}"] option[value="${page.data.origin}"]`)
							$option.selected = true;



						$app.querySelector(`[id="${selector.page}"]`).textContent = page.type


						var $parent = $this.closest(`[class*="${selector.pages}"]`)

						var $labels = $parent.querySelectorAll('label')

						for(var i = 0; i < $labels.length; i++){
							var $label = $labels[i]

							$label.classList.remove(selector.active)
						}


						$this.classList.add(selector.active)


						var $target = $app.querySelector(`[class*="${selector.results}"] [class*="${selector.scroll}"]`)

						$target.innerHTML = ""

						var talks = shadowRoot.querySelector(`[class*="${selector.talks}"]`)

						talks.innerHTML = ''

						app.pages = {}


						$app.classList.add(selector.loading)

						timeout.clear()

						await timeout.fn()

						$app.classList.remove(selector.loading)

						return


					}else if(id.indexOf('team-') > -1){
						id = id.replace('team-',"")

						var user = app.users[id]

						console.log('user',user);

						if(!user){
							return
						}


						if(user.type == "team" && user.from == cookies.address && user.id == user.to){
							console.log('owner 이벤트');
							return
						}

						return

					}else if(id.indexOf('user-') > -1){
						id = id.replace('user-',"")

						if(app.filters.page.id != id){
							var user = app.users[id]

							console.log('page',page);

							

							if(!user){
								return
							}

							app.filters.user = user
							app.filters.type = user.type
							app.filters.created_at = Date.now()


							var $parent = $this.closest(`[class*="${selector.users}"]`)

							var $labels = $parent.querySelectorAll('label')

							for(var i = 0; i < $labels.length; i++){
								var $label = $labels[i]

								$label.classList.remove(selector.active)
							}


							$this.classList.add(selector.active)


							var $target = $app.querySelector(`[class*="${selector.results}"] [class*="${selector.scroll}"]`)

							$target.innerHTML = ""

							var talks = shadowRoot.querySelector(`[class*="${selector.talks}"]`)

							talks.innerHTML = ''


							app.pages = {}


							$app.classList.add(selector.loading)

							timeout.clear()

							await timeout.fn()

							$app.classList.remove(selector.loading)



						}

						return
					}


					let $left = shadowRoot.querySelector(`[id="${selector.left}"]`)
					let $right = shadowRoot.querySelector(`[id="${selector.right}"]`)

					$left.checked = false
					$right.checked = false

				}
			})


			window[`_${selector.profile}`] = async function(e){
				e.preventDefault()

				var name = ""

				if(app.filters.team.type == "team" && app.filters.team.from == cookies.address && app.filters.team.id == user.to && app.filters.team.name == cookies.name){
					name = "Members"
				}


				// prompt("생년월일을 입력해주세요", "1900-11-10");


				return
			}

			window[`_${selector.leave}`] = async function(e){
				e.preventDefault()

				var bool = confirm('Leave ?');

				if(bool){
					var { results } = await app.fetch({
						url : reqUrl( cookies, app.filters, { from: cookies.team, to : cookies.address, href : window.location.href } ),
						method: 'PUT'
					})
				}
				return
			}


			window[`_${selector.invite}`] = async function(e){
				e.preventDefault()

				if(app.block.fetch){
					return
				}

				var $invite = $filters.querySelector(`[class*="${selector.invite}"]`)

				var $email = $invite.querySelector(`[type="email"]`)

				if($email.value){
					app.block.fetch = true

					var { results } = await app.fetch({
						url : reqUrl( cookies, app.filters, { from: cookies.team, to : cookies.address, href : window.location.href } ),
						method: 'PUT'
					})

					if(results.length){
						var invite = results[0]
						
						var $qrcode = $app.querySelector(`[class*="${selector.qrcode}"]`)

						if(!$qrcode.innerHTML){
							new QRCode($qrcode, {
								text: "mailto:"+encodeURIComponent(invite.hook),
								width: 300,
								height: 300,
								colorDark : "#000000",
								colorLight : "#ffffff",
								correctLevel : QRCode.CorrectLevel.H
							})
						}
					}

				}

				return
			}


			var $senderAddress = $app.querySelector(`[name="${selector.address}"]`)

			$senderAddress.addEventListener('submit', async function(e){
				e.preventDefault()

				if(app.block.fetch){
					return
				}

				var $sender = $app.querySelector(`[name="${selector.sender}"]`)

				var sender = $sender.value

				if(sender){
					app.block.fetch = true

					var { cookies } = await app.storage.get('cookies')

					$app.classList.add(selector.loading)

					var { results, session } = await app.fetch({
						url : reqUrl( cookies, app.filters, {sender : sender, href : window.location.href} ),
						method: 'GET',
						headers: {
							"Content-Type": "application/json"
						}
					})

					var $toggle = shadowRoot.querySelector(`[id="${selector.address}"]`)

					$toggle.checked = false

					$sender.value = session.sender

					$app.setAttribute(selector.sender, session.sender ? hashId(session.sender) : "")

					$app.classList.remove(selector.loading)
				}
			})


			var $profile = $app.querySelector(`[class*="${selector.profile}"]`)

			var $setting = $app.querySelector(`form[name="${selector.setting}"]`)

			$setting.addEventListener('submit', async function(e){
				e.preventDefault()

				if(app.block.fetch){
					return
				}

				var $input = $profile.querySelector(`[name="${selector.setting}"] [type="text"]`)

				var $name = $profile.querySelector(`[class*="${selector.name}"] strong`)

				var name = $input.value

				if(name){
					app.block.fetch = true

					var { cookies } = await app.storage.get('cookies')

					$app.classList.add(selector.loading)

					var { results, session } = await app.fetch({
						url : reqUrl( cookies, app.filters, {name : name, href : window.location.href} ),
						method: 'GET',
						headers: {
							"Content-Type": "application/json"
						}
					})

					var $toggle = shadowRoot.querySelector(`[id="${selector.setting}"]`)

					$toggle.checked = false

					$input.value = session.name

					$name.textContent = session.name

					$app.classList.remove(selector.loading)
				}
			})


			var $chat = $app.querySelector(`form[name="${selector.chat}"]`)

			$chat.addEventListener('submit', async function(e){
				e.preventDefault()

				var $message = $chat.querySelector(`[name="${selector.talk}"]`)

				var { cookies } = await app.storage.get('cookies')

				var text = $message.value

				console.log('cookies',cookies);

				console.log('text',text);

				if(text && cookies.address){
					var { cookies } = await app.storage.get('cookies')

					var from = cookies.address
					var to = app.filters.page.id ? app.filters.page.id : hashId(cookies.cc+window.location.pathname)

					$message.disabled = true
					$message.value = "Send..."


					$app.classList.add(selector.loading)

					console.log({ type :'talk', from : from, to : to, text : encodeURIComponent(text) });


					var { results } = await app.fetch({
						url : reqUrl( cookies, app.filters, { type :'talk', from : from, to : to, text : encodeURIComponent(text), href : window.location.href } ),
						method: 'PUT'
					})

					$app.classList.remove(selector.loading)

					$message.disabled = false
					$message.value = ""
				}


				return
			})



			var $prompt = $app.querySelector(`form[name="${selector.prompt}"]`)

			$prompt.addEventListener('submit', async function(e){
				e.preventDefault()

				var { cookies } = await app.storage.get('cookies')

				if(!cookies.sender){
					$app.querySelector(`[for="${selector.address}"]`).click()

					return
				}

				var $context = $prompt.querySelector(`[name="${selector.context}"]`)

				var format = ''

				var body = $context.value ? $context.value : ''

				var href = window.location.href

				if(Object.keys(app.upload).length){
					if(app.upload.format){
						body = app.upload.body
						format = app.upload.format
						href = `https://${app.host}/tracking`
					}
				}


				if(body){
					$app.classList.add(selector.loading)

					$app.setAttribute(selector.block, selector.prompt)

					var query = { from: cookies.address, to : cookies.team, href : href }

					if(format){
						query.format = format
					}

					var req = {
						url : reqUrl( cookies, app.filters, query ),
						method: 'POST',
						headers: {
							'Content-Type': 'application/octet-stream',
							'Content-Encoding': 'gzip'
						}
					}

					if(body){
						req.body = body
					}

					var { results } = await app.fetch(req)

					$app.classList.remove(selector.loading)

					console.log('results',results)

					await Upsert['crons'](results[0])


					// 채팅 추가하는 내용 필요함

					if(!app.chrome){
						var $reset = $app.querySelector(`[id="${selector.reset}"]`)
						var $left = shadowRoot.querySelector(`[id="${selector.left}"]`)
						var $right = shadowRoot.querySelector(`[id="${selector.right}"]`)

						$reset.click()

						$left.checked = false
						$right.checked = true
					}


					if(Object.keys(app.upload).length){
						delete app.upload.format
					}
				}

				return
			})



			

			var $context = $app.querySelector(`[name="${selector.context}"]`)

			$context.placeholder = cookies.hello


			/*
				click event 이후에

				pages.item값이 hide 되면

				tab이벤트가 발생했으니 새로 scan 이벤트 실행해야함

				list selector에서 
			*/

			if(selector.node && selector.item){
				try{
					var $visibleList = Array.from(document.querySelectorAll(selector.item))
						.filter(el => {
							const style = getComputedStyle(el)

							return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0'
						})

					var $before = $visibleList[0]

					if($visibleList.length){
						document.body.addEventListener('click', async function(e){
							var $this = e.target

							var $list = $this.closest(selector.node)

							console.log('selector.node',selector.node);

							if($list){
								if($list.length){
									var $after = $list[0]

									if($before != $after){
										$visibleList = Array.from(document.querySelectorAll(selector.item))
											.filter(el => {
												const style = getComputedStyle(el)

												return style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0'
											})

										if($visibleList.length){
											$before = $after
											
											// 이벤트 시작

										}
									}

								}
							}

						})
					}

				}catch(err){

				}

			}

			if(selector.page && page){
				$app.setAttribute(selector.page, page.type ? page.type : "")

				await timeout.fn()
			}else{
				var $target = $app.querySelector(`[class*="${selector.results}"] [class*="${selector.scroll}"]`)

				if(cookies.address){
					await timeout.fn()
				}else{
					$target.innerHTML = landing.page
				}
			}

			var $sign = {
				in : $app.querySelector(`[id="${selector.sign.in}"]`),
				out : $app.querySelector(`[id="${selector.sign.out}"]`)
			}


			$sign.in.addEventListener('click', async function(e){
				if(!app.chrome){
					var $left = shadowRoot.querySelector(`[id="${selector.left}"]`)
					var $right = shadowRoot.querySelector(`[id="${selector.right}"]`)

					$left.checked = false
					$right.checked = true



					var $qrcode = $app.querySelector(`[class*="${selector.qrcode}"]`)

					if(!$qrcode.innerHTML){
						new QRCode($qrcode, {
							text: "mailto:"+encodeURIComponent(cookies.hash+".logis.center@oauth.email"),
							width: 300,
							height: 300,
							colorDark : "#000000",
							colorLight : "#ffffff",
							correctLevel : QRCode.CorrectLevel.H
						})
					}
				}
			})

			$sign.out.addEventListener('click', async function(e){
				await app.fetch({
					url : reqUrl( cookies, app.filters, {} ),
					method: 'DELETE'
				});

				await app.storage.clear()

				window.location.reload()
			})

			var $home = $app.querySelector(`[id="${landing.home}"]`)

			if($home){
				$home.addEventListener('click', async function(e){
					var $target = $app.querySelector(`[class*="${selector.results}"] [class*="${selector.scroll}"]`)

					if($target.innerHTML != landing.page){
						if(cookies.address){
							// filter 정의 필요

							await timeout.fn()
						}else{
							$target.innerHTML = landing.page
						}
					}
				})
			}


			var $qrauth = $app.querySelector(`[class*="${selector.qrauth}"]`)

			$qrauth.addEventListener('click', onAuth)

			var $qrcode = $app.querySelector(`[class*="${selector.qrcode}"]`)


			console.log('cookiescookiescookiescookies',cookies)

			if(cookies.address){
				$app.setAttribute(selector.address, cookies.address ? cookies.address : "")
				$app.setAttribute(selector.sender, cookies.sender ? hashId(cookies.sender) : "")
			}else{
				console.log(cookies.hash+".logis.center@oauth.email");
				var $qrcode = $app.querySelector(`[class*="${selector.qrcode}"]`)

				if(!$qrcode.querySelector(`img`)){
					new QRCode($qrcode, {
						text: "mailto:"+encodeURIComponent(cookies.hash+".logis.center@oauth.email"),
						width: 300,
						height: 300,
						colorDark : "#000000",
						colorLight : "#ffffff",
						correctLevel : QRCode.CorrectLevel.H
					})
				}
			}

			if(!cookies.address){
				var $target = $app.querySelector(`[class*="${selector.results}"] [class*="${selector.scroll}"]`)

				$target.innerHTML = landing.page
			}


			var canvas = $app.querySelector(`[id="${selector.canvas}"]`);
			var aiocr  = $app.querySelector(`[id="${selector.parse}"]`);
			var photo  = $app.querySelector(`[id="${selector.photo}"]`);

			var cametaStatus = $app.querySelector(`[id="${selector.camera}"]`);

			var capturedBlob = null;


			// "사진 찍기" 버튼 이벤트
			aiocr.addEventListener('click', async function(){
				try{
					var video  = $app.querySelector(`[id="${selector.video}"]`); 

					var context = canvas.getContext('2d');
					
					var videoRatio = video.videoWidth / video.videoHeight;
					var canvasRatio = canvas.width / canvas.height;

					let drawWidth, drawHeight, offsetX, offsetY;

					if (videoRatio > canvasRatio) {
						// 비디오가 캔버스보다 가로가 더 넓은 비율일 경우 (세로를 맞추고 가로를 잘라냄)
						drawHeight = canvas.height;
						drawWidth = drawHeight * videoRatio;
						offsetY = 0;
						offsetX = (canvas.width - drawWidth) / 2; // 중앙에 배치
					} else {
						// 비디오가 캔버스보다 세로가 더 긴 비율일 경우 (가로를 맞추고 세로를 잘라냄)
						drawWidth = canvas.width;
						drawHeight = drawWidth / videoRatio;
						offsetX = 0;
						offsetY = (canvas.height - drawHeight) / 2; // 중앙에 배치
					}

					// 2. 계산된 크기와 위치로 그리기
					context.drawImage(video, offsetX, offsetY, drawWidth, drawHeight);

					canvas.toBlob(async function(blob){
						capturedBlob = blob;
						const imageUrl = URL.createObjectURL(capturedBlob);

						photo.src = imageUrl;
						photo.style.display = 'block';
						cametaStatus.textContent = "사진을 찍었습니다. 전송할 수 있습니다.";

						cametaStatus.textContent = "이미지 압축 및 서버 전송 중...";

						try {
							const imageBuffer = await capturedBlob.arrayBuffer();

							$app.classList.add(selector.loading)

							var body = 'data:image/jpeg;base64,'+bufferToBase64(imageBuffer)


							var created_at = new Date(new Date().getTime() - timezoneOffset).getTime()

							var { cookies } = await app.storage.get('cookies')

							var { results, session } = await app.fetch({
								url : reqUrl( cookies, app.filters, { from: cookies.address, to : cookies.team, href : `https://${app.host}/tracking`, created_at : created_at } ),
								method: 'POST',
								headers: {
									'Content-Type': 'image/jpeg',
									'Content-Encoding': 'gzip' // 데이터가 Gzip 압축되었음을 서버에 알림
								},
								body: body
							});
							
							console.log('서버 응답:', results);
							cametaStatus.textContent = "Completed";
							
						} catch (error) {
							console.error('전송 실패:', error);
							cametaStatus.textContent = "Fail";
						} finally {
							// 전송 완료
						}

					}, 'image/jpeg', 0.85);
				}catch(err){
					console.log('err',err);
				}
					
			});


			var $talks_container = $app.querySelector(`[class*="${selector.talks}"]`)

			$talks_container.addEventListener('click', async function (e) {
				var $this = e.target

				var $talk = $this.closest(`[class*="${selector.talk}"]`) || $this.classList.contains(`[class*="${selector.talk}"]`)

				if($talk){
					var ref = $talk.dataset.ref

					console.log('ref',ref);

					var page = app.pages[ref]

					if(!page){
						var items = await Select['items']({
							key : 'id',
							value : $talk.id
						})

						console.log('items',items);

						if(items.length){
							var item = items[0]

							var pages = await Select['pages']({
								key : 'bcc',
								value : item.bcc
							})

							if(pages.length){
								page = pages[0]
							}
						}
					}

					if(!page){
						return
					}

					let $left = shadowRoot.querySelector(`[id="${selector.left}"]`)
					let $right = shadowRoot.querySelector(`[id="${selector.right}"]`)

					$left.checked = false
					$right.checked = false

					var $target = $app.querySelector(`[class*="${selector.results}"] [class*="${selector.scroll}"]`)

					$target.innerHTML = ""

					var talks = shadowRoot.querySelector(`[class*="${selector.talks}"]`)

					talks.innerHTML = ''

					app.pages = {}

					app.filters.page = page

					app.filters.type = page.type

					app.filters.origin = page.data.origin

					app.filters.created_at = Date.now()

					timeout.clear()

					app.block.fetch = true

					$app.classList.add(selector.loading)

					await timeout.fn()

					$app.classList.remove(selector.loading)
				}
			})


			var $talks_scroll = $app.querySelector(`[class*="${selector.chat}"] [class*="${selector.scroll}"]`)
				$talks_scroll.addEventListener('wheel', async function(e) {
					e.preventDefault();

					if(app.block.fetch){
						return
					}

					$talks_scroll.scrollTop -= e.deltaY * 1; 
				});

				$talks_scroll.addEventListener('scroll', async function(e) {
					if(app.block.fetch || app.block.talks){
						return
					}


					var $this = e.target

					var end = $this.scrollHeight - (window.innerHeight + 10)

					var top = $this.scrollTop

					if(top > end){
						var { cookies } = await app.storage.get('cookies')

						var $talks = $app.querySelectorAll(`[class*="${selector.talk}"]`)

						if($talks.length != cookies.talks){
							
							var $created_at = $app.querySelector(`[class*="${selector.talk}"] input[name="${selector.created_at}"]`)

							if($created_at){
								timeout.clear()

								app.filters.created_at = $created_at.value

								app.block.fetch = true

								$app.classList.add(selector.loading)

								await timeout.fn('talks')

								$app.classList.remove(selector.loading)
							}
						}
					}
				});


			var $items_scroll = $app.querySelector(`[class*="${selector.results}"] [class*="${selector.scroll}"]`)
				$items_scroll.addEventListener('scroll', async function(e) {
					if(app.block.fetch || app.block.items){
						return
					}

					var $this = e.target

					var end = $this.scrollHeight - (window.innerHeight + 100)

					var top = $this.scrollTop

					if(top > end){
						var { cookies } = await app.storage.get('cookies')

						var $items = $app.querySelectorAll(`[class*="${selector.result}"]`)
					

						// console.log('cookies[app.filters.type]',cookies[app.filters.type]);
						// console.log('$items.length',$items.length);

						if(cookies[app.filters.page.type] != $items.length){						

							var $created_at = $app.querySelector(`[class*="${selector.result}"]:last-child input[name="${selector.created_at}"]`)

							if($created_at){
								timeout.clear()

								$app.classList.add(selector.loading)

								app.filters.created_at = $created_at.value

								app.block.fetch = true

								await timeout.fn('items')
							}
						}
					}
				});

				


			var $toggle = $app.querySelector(`[id="${selector.right}"]`)

			$toggle.addEventListener('click', async function(){
				if($toggle.checked){
					isFocus = true

					timeout.clear()

					await timeout.fn()
				}else{
					isFocus = false
					console.log('app.chrome',app.chrome);

					if(!app.chrome){
						shadowRoot.querySelector(`[id="${selector.left}"]`).checked = false
					}

					if(window[cookies.hash]){
						timeout.clear()
					}
				}
			})

			if(!app.chrome){
				await timeout.fn()
			}
		
		}
	}catch(err){
		console.log('err',err);
	}

}())