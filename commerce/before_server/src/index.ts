import { Node, parseHTML } from 'linkedom'

import { gzip, ungzip } from 'pako'

import { ethers } from 'ethers'

import { AwsClient } from 'aws4fetch'

import { renderHtml } from './renderHtml'

/*
	--- 결제 타입 ---
		$user
		$team

		+++ 결제 플로우 만들어야함

	***selector 가 같은데 계속 풀 html 문서 전송막기
*/


var extractNumbersRegex = /\d+/g;


function getZeroUTC(date, day) {
	var date = new Date(date)

	date.setDate(date.getDate() - day)

	date.setUTCHours(0)
	date.setUTCMinutes(0)
	date.setUTCSeconds(0)
	date.setUTCMilliseconds(0)

	return date.getTime() // 'YYYY-MM-DDTHH:mm:ss.sssZ'
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


function hashId(text){
	if(typeof text == "undefined"){
		var account = ethers.Wallet.createRandom()
		text = account.privateKey
	}

	var hashMessage = ethers.hashMessage(text)

	return ethers.computeAddress(hashMessage).toLowerCase()
}


async function Sleep(ms) {
	return new Promise(resolve => setTimeout(resolve, ms))
}


const twoPartDomains = ["co.kr","co.uk","co.jp","com.cn","co.in","com.mx","co.id","com.my","com.sg","com.ph","com.vn"];




/*
	logis 
		- pages 
		- tasks

	사용자 1000명씩 분할
		- vectorize, d1 둘다

	logis-apac1-items
	logis-apac1-goods
	logis-apac1-order
	logis-apac1-tracking
	logis-apac1-event

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


export default {
	// Method GET, POST
	async fetch(
		request: Request,
		env: Env,
		ctx: ExecutionContext
	): Promise<Response> {
		const s3 = new AwsClient({
			accessKeyId: env.aws_access_key_id,
			secretAccessKey: env.aws_secret_access_key,
			service: 's3',
			region: env.aws_region,
		})

		var {
			// 도시 (예: "San Jose")
			city,
			// 국가 코드 (예: "US")
			country,
			// 국가 이름 (예: "United States")
			countryRegion,
			// 대륙 코드 (예: "NA")
			continent,
			// 위도 (예: "37.33940")
			latitude,
			// 경도 (예: "-121.89496")
			longitude,
			// 시도 (예: "California")
			region,
			// 시도 코드 (예: "CA")
			regionCode,
			// 타임존 (예: "America/Los_Angeles")
			timezone,
			// 우편번호 (예: "95113")
			postalCode,
			// AS 번호 (예: "13335")
			asOrganization,
		} = request.cf;

		// 요청자의 IP 주소
		var ip = request.headers.get('cf-connecting-ip');

		if(!ip){
			ip = request.headers.get('X-Real-IP')
		}

		// 응답 본문에 정보를 포함하여 반환
		var geoInfo = {
			ip,
			city,
			country,
			countryRegion,
			continent,
			latitude,
			longitude,
			region,
			regionCode,
			timezone,
			postalCode,
			asOrganization,
		};

		var FLAG = geoInfo.country.toLowerCase()

		var logisRegion = LogisRegion[FLAG]

		var zoneRegion = ''

		var language = languageCode[FLAG]

		console.log('geoInfo',JSON.stringify(geoInfo))

		try{
			var headers = new Headers()

			var cookies = {}

			var cookiesStr = request.headers.get('Cookie')

			var contentType = request.headers.get("Content-Type")
			var contentEncoding = request.headers.get("Content-Encoding")

			if(cookiesStr){
				cookiesStr.split(';').forEach(cookie => {
					const parts = cookie.split('=')
					if (parts.length === 2) {
						const key = parts[0].trim()
						const value = parts[1].trim()
						cookies[key] = value
					}
				})
			}

			const requestUrl = new URL(request.url)
			const queryParams = requestUrl.searchParams

			var req = {
				url:request.url,
				host:requestUrl.hostname,
				method:request.method,
				query:{},
				body:{}
			}

			if(queryParams){
				if(queryParams.size){
					queryParams.forEach((value, key) => {
						req.query[key] = value
					})
				}
			}

			const acceptLanguageHeader = request.headers.get('Accept-Language')

			let preferredLanguage = "" // 기본값 설정

			if (acceptLanguageHeader) {
				// 2. 가져온 헤더 값을 파싱하여 가장 선호하는 언어를 추출합니다.
				// 보통 첫 번째 항목이 가장 선호하는 언어입니다.
				// 복잡한 파싱 로직 (q 값 고려)은 필요에 따라 추가할 수 있습니다.

				const languages = acceptLanguageHeader.split(',')
				if (languages.length) {
					// 첫 번째 언어 코드 (예: "ko-KR")만 사용하고, q 값이나 추가 정보는 제거
					preferredLanguage = languages[0].split(';')[0].trim()
				}
			}


			var redirect = "https://commerce.logis.center/"

			var origin = decodeURIComponent(req.query.origin ? req.query.origin : "")

			var userAgent = request.headers.get('User-Agent')
			

			if(ip){
				var m = ip.match(/\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b/)

				if(m){
					ip = m[0]
				}
			}

			cookies.hello = Hello[language]

			cookies.flag = FLAG

			console.log('cookies',JSON.stringify(cookies))

			console.log('req.query.referrer',req.query.referrer);
			var href = decodeURIComponent(req.query.href ? req.query.href : "").toLowerCase()

			if(origin && href){
				try{
					// [SAFE-URL] 상대 경로 대응을 위한 URL 파싱 보호
					var url;
					try {
						url = new URL(href);
					} catch (e) {
						url = new URL(href, origin);
					}

					if(href.indexOf(origin) > -1 || url.origin === origin){
						var currentHost = url.host

						var host = currentHost.split('.')

						var isTwoPartDomain = twoPartDomains.some(domain => {
							return currentHost.endsWith(domain);
						});

						var cc = ''

						if(isTwoPartDomain){
							cc = host[host.length-3]+"."+host[host.length-2]+"."+host[host.length-1]
						}else{
							cc = host[host.length-2]+"."+host[host.length-1]
						}

						console.log('cc',cc);

						cookies.cc = hashId(cc)

					}

				}catch(err){
					console.log('err',err);
				}
			}


			var now = Date.now()

			var current = new Date(now).toISOString()

			var balance

			if(cookies.hash || (req.query.hash && req.query.token)){
				try{
					if(cookies.hash && req.query.hash){
						if(cookies.hash != req.query.hash){
							cookies.signature = ""
						}
					}

					if(req.query.hash){
						cookies.hash = req.query.hash
					}

					if(req.query.token){
						cookies.token = req.query.token
					}

					var response = await s3.fetch(`https://${env.aws_bucket}.s3.${env.aws_region}.amazonaws.com/hash/${cookies.hash}`, {
						method:'HEAD'
					})

					if(response.status == 200){
						balance = 0
					}

					if(response.headers.get("vapid")){
						cookies.vapid = true
					}

					if(response.headers.get("subscription")){
						cookies.subscription = true
					}

					var email = response.headers.get("email")

					var phone = response.headers.get("phone")

					var address = response.headers.get("address")

					var token = response.headers.get("token")


					if (response.ok) {
						var headObject = {}

						response.headers.forEach((value, key) => {
							if (key.startsWith('x-amz-meta-')) {
								var metaKey = key.replace('x-amz-meta-', '')
								headObject[metaKey] = value
							}
						})


						if(headObject['email']){
							email = headObject['email']
						}

						if(headObject['phone']){
							phone = headObject['phone']
						}

						if(headObject['address']){
							address = headObject['address']
						}
						
						if(headObject['token']){
							token = headObject['token']
						}

					}

					if(email && token == cookies.token){
						cookies.email = email

						if(phone){
							cookies.phone = phone
						}

						cookies.address = address

						cookies.draft = {}

						cookies.pages = {}


						var { results } = await env[logisRegion].prepare(`SELECT * FROM users WHERE "type" = 'user' AND "id" = '${cookies.address}' AND "from" = '${cookies.address}' LIMIT 1`).all()

						var teamId = hashId(cookies.address)

						if(results.length){
							var user = results[0]

							cookies.team = user.to

							var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(user.data))

							var data = JSON.parse(decompressedJsonString)

							cookies.name = data.name

							FLAG = data.flag

							zoneRegion = data.region

							var team = req.query.team ? decodeURIComponent(req.query.team) : data.team
							var name = req.query.name ? decodeURIComponent(req.query.name) : data.name
							var sender = req.query.sender ? decodeURIComponent(req.query.sender) : data.sender



							if(name){
								data.name = cookies.name = name
							}

							if(team){
								data.team = team
							}

							if(sender){
								data.sender = sender
							}


							if(data.sender){
								cookies.sender = data.sender
							}


							user.cc = cookies.cc

							user.ref = hashId(cookies.team+cookies.cc+link)

							var user_arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify(data)), { to: 'arraybuffer' })

							user.data = user_arr.buffer

							var { results, success, error } = await env[logisRegion].prepare(`
								INSERT INTO users ("id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "updated_at") 
								VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10) ON CONFLICT (id) DO UPDATE SET
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
								user.id, user.type, user.from, user.to, user.cc, user.bcc, user.ref, user.data, user.created_at, now
							).run()

						}else{
							var name = cookies.email.split("@")[0]

							var domains = logisRegion.split("_")

							var regionCode = domains[2]


							var { results } = await env[CenterRegion].prepare(`SELECT * FROM zones`).all()

							// 분할 저장

							var pool;


							console.log('regionCode',regionCode);

							for(var z = 0; z < results.length; z++){
								var zone = results[z]

								try{
									var zoneCode = zone.id.split(extractNumbersRegex)[0]

									if(typeof pool == "undefined"){
										if(regionCode == zoneCode){
											if(zone.pool < 1000){
												zoneRegion = zone.id
												pool = zone.pool
											}
										}
									}
								}catch(err){
									console.log('zoneRegion err',err);
								}
							}


							var user_arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
								flag : FLAG,
								name : name,
								title : "",
								region : zoneRegion,
								page_count : 0,
								favicon : null
							})), { to: 'arraybuffer' })

							var user = {
								id : cookies.address,
								type : "user",
								from : cookies.address,
								to : cookies.team ? cookies.team : teamId,
								cc  : "",
								bcc : "",
								ref  : "",
								data : user_arr.buffer,
								created_at : now,
								updated_at : now
							}


							var base = {
								pages : {},
								goods : {
									draft : 0,
									count : 0
								},
								order : {
									draft : 0,
									count : 0
								},
								event : {
									draft : 0,
									count : 0
								},
								coupon : {
									draft : 0,
									count : 0
								},
								tracking : {
									draft : 0,
									count : 0
								},
								search : {
									draft : 0,
									count : 0
								},
								review : {
									draft : 0,
									count : 0
								},
								member : {
									draft : 0,
									count : 0
								}
							}

							var propertys = ['price', 'quantity', 'width', 'height', 'length', 'weight', 'shipping_fee', 'shipping_duration', 'sale_price', 'supply_price', 'low_stock_threshold', 'discount', 'min_order_amount', 'max_discount_amount', 'usage_limit', 'usage_per', 'started_at', 'expired_at']

							for(var p = 0; p < propertys.length; p++){
								var property = propertys[p]

								for (const table in base) {
									if (base.hasOwnProperty(table)) {
										if(table != "pages"){
											base[table][property] = {
												max : 0,
												min : 0
											}
										}
									}
								}
							}

							var team_arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
								flag : FLAG,
								name : name+'\'s team',
								title : "",
								region : zoneRegion,
								page_count : 0,
								favicon : null,
								base : base
							})), { to: 'arraybuffer' })


							

							var team = {
								id : teamId,
								type : "team",
								from : cookies.address,
								to : teamId,
								cc : "", // 조직 도메인 참조
								bcc : "", // // 상위 조직 참조시 사용
								ref : "", 
								data : team_arr.buffer,
								created_at : now,
								updated_at : now
							}

							cookies.name = name

							cookies.team = team.id



							var statements = []

							statements.push(
								env[logisRegion].prepare(`
									INSERT INTO users ("id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "updated_at") 
									VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
								`).bind(
									team.id, team.type, team.from, team.to, team.cc, team.bcc, team.ref, team.data, team.created_at, team.updated_at
								)
							)

							statements.push(
								env[logisRegion].prepare(`
									INSERT INTO users ("id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "updated_at") 
									VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10) ON CONFLICT (id) DO NOTHING
								`).bind(
									user.id, user.type, user.from, user.to, user.cc, user.bcc, user.ref, user.data, user.created_at, user.updated_at
								)
							)

							try{
								await env[logisRegion].batch(statements)

								await env[CenterRegion].prepare(`
									INSERT INTO zones ("id", "pool") 
									VALUES (?1, ?2)
									ON CONFLICT(id) DO UPDATE SET
										"pool" = EXCLUDED."pool"
								`).bind(
									zoneRegion,
									pool+1
								).run()
							}catch(err){

							}
						}

						try{
							var clientAddress = ethers.hashMessage(email)
								clientAddress = ethers.computeAddress(clientAddress).toLowerCase()

							var response = await s3.fetch(`https://${env.aws_bucket}.s3.${env.aws_region}.amazonaws.com/address/${clientAddress}`, {
								method:'HEAD'
							})

							

							if (response.status == 200) {
								cookies.balance = balance = response.headers.get("balance") * 1
							}

							if (response.ok) {
								var headObject = {}

								response.headers.forEach((value, key) => {
									if (key.startsWith('x-amz-meta-')) {
										var metaKey = key.replace('x-amz-meta-', '')
										headObject[metaKey] = value
									}
								})


								if(headObject['balance']){
									cookies.balance = balance = headObject['balance'] * 1
								}
							}
						}catch(err){
							
						}

						
					}else{
						headObject = undefined
					}

					headers.set('Set-Cookie', `hash=${cookies.hash}; Secure; HttpOnly; SameSite=None`)
					headers.append('Set-Cookie', `token=${cookies.token}; Secure; HttpOnly; SameSite=None`)

					if(cookies.team){
						headers.append('Set-Cookie', `team=${cookies.team}; Secure; HttpOnly; SameSite=None`)
					}
				}catch(err){
					console.log("errrrr",err)
					cookies = {err : JSON.stringify(err), logisRegion : logisRegion, zoneRegion : zoneRegion}
				}
			}

			console.log('zoneRegion',zoneRegion);

			if(typeof balance == "undefined"){
				var account = ethers.Wallet.createRandom()
				var hash = account.address.toLowerCase()
					hash = hash.replace("0x","")

				var token = account.privateKey.toLowerCase()
					token = token.replace("0x","")

				try{
					var response = await fetch(
						await s3.sign(`https://${env.aws_bucket}.s3.${env.aws_region}.amazonaws.com/hash/${hash}`, {
							method:'PUT',
							headers:{
								'Content-Type': 'text/plain',
								...Object.fromEntries(
									Object.entries({
										balance: '1000',
										host: 'logis.center',
										token: token,
										email: '',
										ip: ip,
										phone: '',
										address: '',
									}).map(([key, value]) => [`x-amz-meta-${key}`, value])
								),
							},
							body:"",
							aws: {
								service: 's3',
								region: env.aws_region,
							}
						})
					)

					cookies.token = token
					cookies.hash = hash

					headers.set('Set-Cookie', `hash=${hash}; Secure; HttpOnly; SameSite=None`)
					headers.append('Set-Cookie', `token=${token}; Secure; HttpOnly; SameSite=None`)
				}catch(err){
					cookies.err = JSON.stringify(err)
					console.log("erraasd", err)
				}
			}


			headers.set('Access-Control-Allow-Origin', '*')
			headers.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
			headers.set('Access-Control-Allow-Headers', 'Content-Type, Content-Encoding, Authorization, X-Requested-With')

			headers.set('Content-Security-Policy', `frame-ancestors 'self' ${origin};`)




			console.log('request.method',request.method)
			console.log('contentEncoding',contentEncoding)

			var created_at = 0

			if(!isNaN(req.query.created_at)){
				created_at = parseInt(req.query.created_at)
			}

			// [SAFE-URL] 상대 경로 대응을 위한 URL 파싱 보호
			var url;
			try {
				url = new URL(href);
			} catch (e) {
				url = new URL(href, origin || "https://commerce.logis.center");
			}

			var pathname = url.pathname.toLowerCase()

			var link = (url.pathname+url.search).toLowerCase()


			var referrer

			try{
				if(req.query.referrer){
					var _url = new URL(req.query.referrer)

					var { results } = await env[CenterRegion].prepare(`SELECT * FROM pages WHERE "id" = '${hashId(cookies.cc+_url.pathname.toLowerCase())}' AND "created_at" < ${created_at} LIMIT 1`).all()

					if(results.length){
						referrer = results[0]
						referrer.table = 'pages'

					}
				}

			}catch(err){
				console.log('req.query.referrer err',err);
			}



			var pageId = ethers.isAddress(req.query.id) ? req.query.id.toLowerCase() : hashId(cookies.cc+pathname)

			var itemId = hashId(cookies.team+cookies.cc+link)

			

			var homeCc = hashId('logis.center')

			var homeId = hashId(homeCc.toUpperCase()+'/tracking')

			var homeBcc = hashId('tracking'+homeCc.toUpperCase())

			var homeRef = hashId(cookies.team+homeCc+'/tracking')


			var home = {
				id : homeId,
				table : 'pages',
				type : 'tracking',
				from : cookies.address ? cookies.address : ethers.ZeroAddress,
				to : cookies.team ? cookies.team : ethers.ZeroAddress,
				cc : homeCc,
				bcc : homeBcc,
				ref : homeRef,
				data : {
					type : 'tracking',
					origin : "https://logis.center",
					link : '/tracking',
					item : true,
					node : true
				},
				created_at : now,
				updated_at : now
			}

			home.data = gzip(new TextEncoder('utf-8').encode(JSON.stringify(home.data)), { to: 'arraybuffer' })




			var tasks = []

			var items = []

			var hints = []

			var detail = {}

			var pages = [home]

			var page = homeId == pageId ? home : {}

			var isDetail = false

			var temp = {}

			var type = homeId == pageId ? 'tracking' : ''



			// console.log('homeId == pageId',homeId == pageId ? 'true' : 'false');

			var member = false

			var users = []

			var admin = {}

			var { results, success, error } = await env[logisRegion].prepare(
				`SELECT * FROM users WHERE "to" = '${cookies.team}' AND "created_at" < ${created_at} ORDER BY created_at DESC LIMIT 10`
			).all()

			if(results.length){
				for(var i = 0; i < results.length; i++){
					var user = results[i]
					
					user.table = 'users'

					var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(user.data))

					var data = JSON.parse(decompressedJsonString)

					if(data.name){
						user.name = data.name
					}

					delete user.data

					if(user.id == user.to){
						admin = user

						member = true

						cookies.pages = data.base.pages
						
						cookies.goods = data.base.goods.count
						cookies.order = data.base.order.count
						cookies.tracking = data.base.tracking.count
						cookies.event = data.base.event.count
						cookies.coupon = data.base.event.coupon
					}

					users.push(user)
				}
				
			}


			cookies.type = req.query.type

			if(cookies.address){

				var { results } = await env[`commerce_logis_${zoneRegion}_items`].prepare(`SELECT * FROM items WHERE "to" = '${cookies.team}' AND "ref" = '${itemId}' LIMIT 1`).all()

				// console.log('detail results.length',results.length);
				// console.log(`SELECT * FROM items WHERE "to" = '${cookies.team}' AND "ref" = '${itemId}' LIMIT 1`);

				if(results.length){
					detail = results[0]

					items.push(detail)
				}

				console.log('detail',JSON.stringify(detail))

				var { results } = await env[`commerce_logis_${zoneRegion}_items`].prepare(`SELECT * FROM items WHERE "type" = 'pages' AND "to" = '${cookies.team}' LIMIT 100`).all()

				console.log('page results.length',results.length);

				if(results.length){
					for(var i = 0; i < results.length; i++){
						var _page = results[i]

						var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(_page.data))

						var page_data = JSON.parse(decompressedJsonString)

						console.log('page_data.type',page_data.type);

						_page.type = page_data.type
						_page.from = cookies.address ? cookies.address : ethers.ZeroAddress
						_page.to = cookies.team ? cookies.team : ethers.ZeroAddress
						
						_page.table = 'pages'

						temp[_page.id] = true

						// page.bcc = hashId(page.type+(isDetail ? task.cc.toUpperCase() : task.cc))


						if(detail.bcc == _page.bcc){
							_page.current = true
							page = _page

							if(_page.type == "tracking"){
								type = "tracking"

							}else if(_page.type == "goods" || _page.type == "order"){
								type = "sales"

							}else if(_page.type == "event" || _page.type == "coupon"){
								type = "event"

							}

							pageId = page.id


							isDetail = true
						}

						if(Object.keys(detail).length == 0 && pageId == _page.id){
							_page.current = true
							page = _page

							if(_page.type == "tracking"){
								type = "tracking"

							}else if(_page.type == "goods" || _page.type == "order"){
								type = "sales"

							}else if(_page.type == "event" || _page.type == "coupon"){
								type = "event"

							}

							if(Object.keys(page_data).length){
								isDetail = page_data.detail
							}
						}


						pages.push(_page)

					}
				}



				
					




				// if(pages.length){
				// 	var condition = ``
				// 	for(var p = 0; p < pages.length; p++){
				// 		var _page = pages[p]

				// 		if(condition){
				// 			condition ++ ' OR '
				// 		}

				// 		condition = ` ("id" = '${_page.id}' AND "created_at" < ${created_at}) `
				// 	}

				// 	var { results } = await env[CenterRegion].prepare(`SELECT * FROM pages WHERE ${condition} LIMIT 100`).all()

				// 	if(results.length){
				// 		for(var r = 0; r < results.length; r++){
				// 			var before = results[r]

				// 			for(var p = 0; p < pages.length; p++){
				// 				var after = pages[p]

				// 				if(before.id == after.id){
				// 					pages[p] = before
				// 				}
				// 			}
				// 		}
				// 	}
				// }

				console.log('Object.keys(page).length',Object.keys(page).length);

				if(Object.keys(page).length == 0){
					var { results } = await env[`commerce_logis_${zoneRegion}_items`].prepare(`SELECT * FROM items WHERE "type" = 'pages' AND "cc" = '${cookies.cc}' AND "created_at" < ${created_at} LIMIT 100`).all()

					if(results.length){
						var statements = []

						for(var i = 0; i < results.length; i++){
							var _page = results[i]

							if(temp[_page.id]){
								continue;
							}

							temp[_page.id] = true

							var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(_page.data))

							var page_data = JSON.parse(decompressedJsonString)

							var bool = cookies.team == _page.to

							_page.type = page_data.type
							_page.from = cookies.address ? cookies.address : ethers.ZeroAddress
							_page.to = cookies.team ? cookies.team : ethers.ZeroAddress

							_page.table = 'pages'

							if(detail.bcc == _page.bcc){
								_page.current = true
								page = _page

								if(_page.type == "tracking"){
									type = "tracking"

								}else if(_page.type == "goods" || _page.type == "order"){
									type = "sales"

								}else if(_page.type == "event" || _page.type == "coupon"){
									type = "event"

								}

								pageId = page.id


								isDetail = true
							}


							if(Object.keys(detail).length == 0 && pageId == _page.id){
								_page.current = true
								page = _page

								if(_page.type == "tracking"){
									type = "tracking"

								}else if(_page.type == "goods" || _page.type == "order"){
									type = "sales"

								}else if(_page.type == "event" || _page.type == "coupon"){
									type = "event"

								}


								if(Object.keys(_page).length){
									isDetail = page_data.detail
								}
							}

							var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify(page_data)), { to: 'arraybuffer' })


							if(!bool){
								statements.push(
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
										_page.id,
										'pages',
										_page.from,
										_page.to,
										_page.cc,
										_page.bcc,
										_page.ref,
										_page.digest,
										arr.buffer,
										now,
										now
									)
								)
							}

							_page.data = arr

							pages.push(_page) 

								
						}

						if(statements.length){
							await env[`commerce_logis_${zoneRegion}_items`].batch(statements)
						}
					}else{
						var { results } = await env[CenterRegion].prepare(`SELECT * FROM pages WHERE "id" = '${pageId}' AND "created_at" < ${created_at} LIMIT 1`).all()

						if(results.length){
							page = results[0]

							var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(page.data))

							var page_data = JSON.parse(decompressedJsonString)

							page.current = true

							page.from = cookies.address ? cookies.address : ethers.ZeroAddress
							page.to = cookies.team ? cookies.team : ethers.ZeroAddress
							page.table = 'pages'

							if(page.type == "goods" || page.type == "order"){
								type = "sales"

							}else if(page.type == "event" || page.type == "coupon"){
								type = "event"

							}

							temp[page.id] = page

							pages.push(page)


							if(Object.keys(page).length){
								isDetail = page_data.detail
							}
						}
					}
						
				}





				if(req.query.crons){
					try{
						var crons = JSON.parse(req.query.crons)

						if(crons.length){
							var cron_condition = ''

							var tasks = []

							var temp = {}

							for(var c = 0; c < crons.length; c++){
								var cron = crons[c].toLowerCase()

								if(ethers.isAddress(cron)){
									cron_condition += ` OR ("id" = '${cron}' AND "from" = '${cookies.address}' AND "created_at" < ${now}) `

									temp[cron] = c

									tasks.push({
										id : cron,
										table : 'tasks',
										type : '',
										from : '',
										to : '',
										cc : '',
										bcc : '',
										ref : '',
										data : null,
										created_at : now,
										updated_at : now
									})
								}
							}

							cron_condition = cron_condition.replace(' OR ','')

							var { results, success } = await env[logisRegion].prepare(`SELECT * FROM tasks WHERE ${cron_condition} ORDER BY created_at LIMIT ${crons.length}`).all()
							
							if(results.length){
								for(var i = 0; i < results.length; i++){
									var cron = results[i]

									var index = temp[cron.id]

									if(typeof index != "undefined"){
										tasks.splice(index, 1);
									}
								}
							}
						}
					}catch(err){
						console.log('tasks err',err);
					}		
				}
					
				
			}
			

			if(pages.length){
				try{
					var branchs = []

					var _pages = safeClone(pages)

					for (var p = 0; p < _pages.length; p++) {
						var _page = _pages[p]

						var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(_page.data))

						_page.data = JSON.parse(decompressedJsonString)

						var _url = new URL(_page.data.origin)

						if(_page.data.item){
							branchs[`${_page.data.origin}#${_page.type}`] = { ..._page, children: [] }
						}

						branchs[_page.id] = { ..._page, children: [] }
					}


					var _temp = {}

					for(var key in branchs){
						if (branchs.hasOwnProperty(key)) {
							var _page = safeClone(branchs[key])

							if(!_temp[_page.id]){
								_temp[_page.id] = true

								var parent = branchs[`${_page.data.origin}#${_page.type}`]

								if(parent){
									if(_page.data.item){
										var children = safeClone(parent.children)

										branchs[`${_page.data.origin}#${_page.type}`] = {
											..._page,
											children : children
										}
									}else{
										branchs[`${_page.data.origin}#${_page.type}`].children.push(_page)
									}
									
								}else{
									if(_page.data.item){
										if(!branchs[`${_page.data.origin}#${_page.type}`]){
											branchs[`${_page.data.origin}#${_page.type}`] = {
												..._page,
												children : []
											}
										}
									}else{
										branchs[`${_page.data.origin}#${_page.type}`].children.push(_page)
									}
								}


										
							}
						}
					}

					// console.log('branchs',JSON.stringify(branchs));

					var tree = []

					for(var key in branchs){
						if (branchs.hasOwnProperty(key)) {
							if(!ethers.isAddress(key)){
								var _page = branchs[key]

								tree.push(_page)
							}
						}
					}

					// console.log('tree',JSON.stringify(tree));	
				}catch(err){

				}

			}


			cookies.bcc = ''

			if(Object.keys(page).length){
				cookies.bcc = hashId(page.type+(isDetail ? cookies.cc.toUpperCase() : cookies.cc))

				if(!member){
					var { results } = await env[`commerce_logis_${zoneRegion}_items`].prepare(`SELECT * FROM items WHERE "type" = 'member' AND "from" = '${cookies.address}' AND "to" = '${cookies.team}' AND "bcc" = '${cookies.bcc}' LIMIT 1`).all()

					if(results.length){
						member = true
					}else{
						var { results } = await env[`commerce_logis_${zoneRegion}_items`].prepare(`SELECT * FROM items WHERE "type" = 'member' AND "from" = '${cookies.address}' AND "to" = '${cookies.team}' AND "bcc" = '${cookies.cc}' LIMIT 1`).all()
						
						if(results.length){
							member = true
						}
					}
				}
			}


			cookies.href = href

			/*
				page 매칭 2개 가능
					id = hashId(link) // 가장 정확

					cc = hashId(cookies.cc) // 부분 매칭 가능
			*/


			console.log('pageId',pageId);


			if(request.method == 'PUT'){
				/*
					초대팀 id
						보내는 사람
							team id + 초대받는 id
							var id = hashId(team id + 초대받는 id)

						받는 사람


					plan에 따라 초대 가능 인원 제한하기

				*/

				var success = false

				var results = []

				if(cookies.sender){
					if(ethers.isAddress(req.query.from) && ethers.isAddress(req.query.to)){

						var refs = []

						try{
							if(req.query.ref){
								refs = JSON.parse(req.query.ref)
							}
						}catch(err){

						}

						if(ethers.isAddress(req.query.id) && refs.length){
							var statements = []

							for(var r = 0; r < refs.length; r++){
								var ref = refs[r]

								if(req.query.cc == ethers.ZeroAddress){
									/*
										나가기 or 내보내기
											ref[] 값으로 내보내기
									*/ 

									if(ethers.isAddress(req.query.to) && req.query.id == cookies.team){
										// 권한 해제

										if(team.from != req.query.to){
											var { results } = await env[`commerce_logis_${zoneRegion}_items`].prepare(
												`SELECT * FROM items WHERE "id" = ?1 AND "type" = ?2 AND "from" = ?3 AND "to" = ?4 AND "bcc" = ?5 ORDER BY created_at DESC`
											).bind(req.query.id, "member", cookies.address, req.query.to, cookies.address).all()

											if(results.length){
												var row = results[0]

												row.to = ethers.ZeroAddress

												statements.push(
													env[`commerce_logis_${zoneRegion}_items`].prepare(
														`UPDATE items SET to = ?, updated_at = ? WHERE id = ?`
													).bind(
														row.to, now, row.id
													)
												)
											}
										}

										// 해당 팀원이 본인 팀으로 되어있으면 비우기
									}

								}else if(ethers.isAddress(req.query.cc)){
									/*
										허용
											ref[] 값으로 허용
									*/ 

									if(req.query.from == cookies.address && req.query.id == cookies.team){
										if(team.from != req.query.to){
											var memberId = hashId(cookies.address+req.query.to+cookies.team)

											var { results, success, error } = await env[`commerce_logis_${zoneRegion}_items`].prepare(
												`SELECT * FROM items WHERE "id" = '${memberId}' AND "type" = 'member' ORDER BY created_at DESC`
											).all()

											if(results.length){
												
											}else{
												var { results, success, error } = await env[`commerce_logis_${zoneRegion}_items`].prepare(
													`SELECT * FROM items WHERE "id" = '${memberId}' AND "from" = '${req.query.to}' AND "cc" = '${req.query.cc}' ORDER BY created_at DESC`
												).all()

												if(results.length){
													var row = results[0]

													var member = {
														id : hashId(cookies.address+row.from+cookies.team),
														type : "member",
														from : cookies.address,
														to : row.from,
														cc : cookies.team,
														bcc : cookies.address,
														ref : "",
														data : null,
														created_at : now,
														updated_at : now
													}

													var name = req.query.to.split("@")[0]

													var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
														flag : FLAG,
														name : name,
														title : "",
														region : null,
														page_count : 0,
														favicon : null
													})), { to: 'arraybuffer' })


													statements.push(
														env[`commerce_logis_${zoneRegion}_items`].prepare(`
															INSERT INTO items ("id", "type", "from", "to", "cc", "bcc", "ref", "digest", "data", "created_at", "updated_at") 
															VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
															ON CONFLICT(id) DO UPDATE SET
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
															member.id,
															member.type,
															member.from,
															member.to,
															member.cc,
															member.bcc,
															member.ref,
															member.digest,
															arr.buffer,
															member.created_at,
															member.updated_at
														)
													)
												}
											}
										}

											
									}else if(req.query.from == cookies.team && req.query.to == cookies.address){
										/*
											참여
												req.query.from 팀장
												
										*/

										// var { results, success, error } = await env[`commerce_logis_${zoneRegion}_items`].prepare(
										// 	`SELECT * FROM items WHERE "id" = ?1 AND "to" = ?2 AND "cc" = ?3 AND "ref" = ?3 ORDER BY created_at DESC`
										// ).bind(req.query.id, cookies.address, req.query.cc, "").all()

										// if(results.length){
										// 	var row = results[0]

										// 	try{
										// 		var { results, success, error } = await env[logisRegion].prepare(`
										// 			UPDATE users SET to = ? WHERE id = ?
										// 		`).bind(
										// 			row.cc, cookies.address
										// 		).run()

										// 	}catch(err){
										// 		console.log('user insert err',err)
										// 		results = []
										// 	}
										// }
									}
								}
							}


							try{
								await env[`commerce_logis_${zoneRegion}_items`].batch(statements)
							}catch(err){

							}



						}else if(req.query.text){

							var text = decodeURIComponent(req.query.text)

							var url = new URL(cookies.href)

							var link = (url.pathname + url.search).toLowerCase()

							var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
								text : text,
								link : link,
								origin :  url.origin
							})), { to: 'arraybuffer' })


							var talk = {
								id : hashId(),
								type : 'talk',
								from : req.query.from,
								to : req.query.to,
								cc : cookies.cc,
								bcc: cookies.bcc,
								ref : hashId(cookies.team+(isDetail ? cookies.cc.toUpperCase() : cookies.cc)+link),
								data : arr.buffer,
								created_at : now,
								updated_at : 0
							}


							var { results, success, error } = await env[`commerce_logis_${zoneRegion}_talks`].prepare(`
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
								talk.created_at,
								talk.updated_at
							).run()
						}

					}
				}


				headers.set('Content-Type', 'application/json')

				return new Response(JSON.stringify({
					results:results,
					success:success,
					session:cookies
				}), {
					headers:headers
				})

			}else if(request.method == 'DELETE') {
				if(cookies.address){
					if(req.query.type == 'crons'){
						if(ethers.isAddress(req.query.ref)){
							var ref = req.query.ref.toLowerCase()

							var { results, success, error } = await env[`commerce_logis_${zoneRegion}_items`].prepare(
								`DELETE FROM tasks WHERE ref = "${ref}"`
							).run()
						}
					}else{
						var response = await fetch(
							await s3.sign(`https://${env.aws_bucket}.s3.${env.aws_region}.amazonaws.com/hash/${cookies.hash}`, {
								method:'DELETE'
							})
						)

						console.log('response',JSON.stringify(response));

						cookies = {}
					}
				}


				return new Response(JSON.stringify({
					success:false,
					results:[],
					session:cookies
				}), {
					headers:headers
				})



			}else if(request.method == 'POST') {
				if(cookies.sender && created_at){
					var task = {
						cc:cookies.cc,
						ref : '',
						text:"",
						digest : ''
					}


					// tracking의 경우 별도 페이지 값 가져와야함

					try{
						if(req.query.referrer){
							var _url = new URL(decodeURIComponent(req.query.referrer))
							
							task.referrer = hashId(cookies.cc+_url.pathname.toLowerCase())
						}

					}catch(err){
						console.log("task req.query.referrer err",err);
					}
					
					try{
						if(contentEncoding == "gzip"){
							var buffer = await request.arrayBuffer()

							console.log('buffer.byteLength',buffer.byteLength);

							console.log('req.query',JSON.stringify(req.query));
							console.log('contentType',contentType);

							if(buffer.byteLength){
								if(req.query.from == cookies.address && req.query.to == cookies.team){

									task.from = cookies.address
									task.to = cookies.team


									var body = new TextDecoder('utf-8').decode(ungzip(buffer))


									if(req.query.format){
										task.body = body

										task.scan = true
									}else{
										task.body = decodeURIComponent(body)
									}



									task.contentType = contentType

									// if(Object.keys(page).length){ // 페이지 스캔 여부 확인
									// 	try{
									// 		var { document } = parseHTML(`<html><body>${body}</body></html>`);

									// 		var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(page.data))

									// 		var selectors = JSON.parse(decompressedJsonString)

									// 		if(page.type){
									// 			var results = []

									// 			for (const s in selectors) {
									// 				if (selectors.hasOwnProperty(s)) {
									// 					var selector = selectors[s]

									// 					var item = {}

									// 					var $item = document.querySelector(selector)

									// 					if($item){
									// 						var checked = $item.getAttribute('checked')

									// 						var selected = $item.getAttribute('selected')

									// 						if($item.getAttribute('type')){
									// 							var text = $item.textContent

									// 							if(checked){
									// 								item[s] = checked == "true" ? true : false
									// 							}else if(selected){
									// 								item[s] = selected
									// 							}else if($item.value){
									// 								item[s] = $item.value
									// 							}else if($item.textContent){
									// 								item[s] = $item.textContent		
									// 							}else{
									// 								item[s] = null
									// 							}
									// 						}else{
									// 							item[s] = $item.textContent ? $item.textContent : null	
									// 						}
									// 					}

									// 					results.push(item)
									// 				}
									// 			}

									// 			headers.set('Content-Type', 'application/json')

									// 			return new Response(JSON.stringify({
									// 				results:results,
									// 				success:success,
									// 				session:cookies
									// 			}), {
									// 				headers:headers
									// 			})
									// 		}

									// 	}catch(err){
									// 		console.log('selectors err',err);
									// 	}
									// }

								}else{
									var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(buffer))

									var data = JSON.parse(decompressedJsonString)

									if(data.length){
										if(typeof data == "object"){
											var list = data

											var condition = ""

											var url = new URL(cookies.href)

											var link = (url.pathname + url.search).toLowerCase()

											var ref = hashId(cookies.team+cookies.cc+link)

											for(var i = 0; i < list.length; i++){
												try{
													var url = new URL(list[i])

													var link = (url.pathname + url.search).toLowerCase()

													if(condition){
														condition += ' OR '
													}

													condition += `("to" = "${cookies.team}" AND "bcc" = '${hashId(page.type+cookies.cc.toUpperCase())}' AND "ref" = '${hashId(cookies.team+cookies.cc+link)}' AND "created_at" < ${created_at})`

												}catch(err){

												}
											}

											var success = false

											var results = []

											console.log('condition',condition);

											if(condition){
												var { results, success, error } = await env[`commerce_logis_${zoneRegion}_items`].prepare(
													`SELECT * FROM items WHERE ${condition} ORDER BY created_at DESC LIMIT 1000`
												).all()

												// if(results.length){
												// 	var item = results[0]

												// 	var { results, success, error } = await env[`commerce_logis_${zoneRegion}_items`].prepare(
												// 		`SELECT * FROM ${item.type} WHERE ${condition} ORDER BY created_at DESC LIMIT 100`
												// 	).all()
												// }
											}

											headers.set('Content-Type', 'application/json')

											return new Response(JSON.stringify({
												success:success,
												results:results,
												session:cookies
											}), {
												headers:headers
											})
										}else if(typeof data == "string"){
											if(Object.keys(page).length){
												task.page = page
												task.from = cookies.team
												task.to = cookies.address
												task.body = data.trim()
												task.scan = null
											}else{
												// 질의를 하기위한 정보가 없음
												return new Response(JSON.stringify({
													success:false,
													results:[],
													session:cookies
												}), {
													headers:headers
												})
											}
										}
									}

								}
							}
						}
					}catch(err){
						console.log('gzip or json err',err)
					}


					console.log('task.from, task.to', task.from, task.to)

					console.log('zoneRegion', zoneRegion)

					console.log('typeof task.body', typeof task.body)


					


					try {
						if(task.from && task.to && task.body){
							task.no = ""

							var url = new URL(cookies.href)

							task.origin = url.origin

							task.link = (url.pathname + url.search).toLowerCase()

							task.ref = hashId(cookies.team+task.cc+url.pathname)

							task.href = cookies.href

							task.type = "prompt"

							task.title = req.query.title ? decodeURIComponent(req.query.title) : ""

							task.topK = 50

							task.rpm = 3

							task.from = cookies.address

							task.to = cookies.team

							task.bcc = ""

							task.contentType = task.contentType


							console.log('req.query.format',req.query.format);

							if(req.query.format){
								if(req.query.format.indexOf("image/") > -1){
									task.type = "tracking"

									task.bcc = hashId(task.type+task.cc.toUpperCase())

									task.contentType = decodeURIComponent(req.query.format)
								}
							}


							console.log('task.type',task.type);

							


							task.flag = FLAG

							task.score = 0
							task.amount = 0

							task.quantity = 0

							task.currency = ""

							task.semantic = ""

							task.zone = zoneRegion


							/*
								link 이미 있으면 참조하기

								id 생성 text가 아닌 req.query.referrer 만들기
							*/

							task.id = hashId(cookies.team+task.cc+task.body)

							var current_date = current.split("T")
								current_date = current_date[0].split("-")

							var year = current_date[0]

							var month = current_date[1]

							var $to = hashId(year+"-"+month)

							var { results } = await env[`commerce_logis_${zoneRegion}_items`].prepare(`SELECT * FROM items WHERE "type" = 'invoice' AND "from" = '${cookies.address}' AND "to" = '${$to}' AND "created_at" < ${now} LIMIT 1`).all()

							// 보강해야함 - 사용자 결제 혹은 팀결제 확인해서 진행하기

							if(results.length){
								// 유료 사용자
								task.topK = 100
								task.rpm = 50
							}

							task.created_at = now
							task.updated_at = 0

							task.detail = false

							if(Object.keys(page).length){
								task.detail = page.data.detail

								if(task.detail){
									task.ref = hashId(cookies.team+task.cc.toUpperCase()+url.pathname)
								}
							}


							// 한도 초과인지 체크하기

							console.log('cookies.balance',cookies.balance)

							if(cookies.balance){
								headers.set('Content-Type', 'application/json')

								var { results, success } = await env[logisRegion].prepare(`SELECT * FROM tasks WHERE "id" = '${task.id}' AND "created_at" < ${now}  ORDER BY created_at LIMIT 1`).all()

								if(results.length){
									if(contentEncoding == "gzip"){
										if(Object.keys(page).length){
											page.from = cookies.address
											page.to = cookies.team
											page.bcc = cookies.bcc
											page.title = ""
											// page.link = results[0].link
											// page.list = results[0].list
											// page.item = results[0].item

											results = [page]
										}

									}else{
										/*
											text 2 sql 진행
										*/

									}

									return new Response(JSON.stringify({
										results:results,
										success:success,
										session:cookies
									}), {
										headers:headers
									})
								}else{
									var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify(task)), { to: 'arraybuffer' })

									var onConflict = "ON CONFLICT (id) DO NOTHING"

									if(req.query.error){
										onConflict = `ON CONFLICT (id) DO UPDATE SET
											"cc" = EXCLUDED."cc",
											"bcc" = EXCLUDED."bcc",
											"ref" = EXCLUDED."ref",
											"task" = EXCLUDED."task",
											"created_at" = EXCLUDED."created_at",
											"updated_at" = EXCLUDED."updated_at"
										`
									}

									// tasks 추가 작업 
									var { results, success, error } =  await env[logisRegion].prepare(`
										INSERT INTO tasks (
											"id", "type", "from", "to", "cc", "bcc", "ref", "data", "created_at", "updated_at"
										) VALUES (
											?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10
										) ${onConflict}
									`).bind(
										task.id,
										task.type,
										task.from,
										task.to,
										task.cc,
										task.bcc,
										task.ref,
										arr.buffer,
										task.created_at,
										task.updated_at
									).run()

									return new Response(JSON.stringify({
										results:[
											{
												id : task.id,
												table : 'tasks',
												type : task.type,
												from : task.from,
												to : task.to,
												cc : task.cc,
												bcc : task.bcc,
												ref : task.href,
												data : arr,
												created_at : task.created_at,
												updated_at : task.updated_at
											}
										],
										success:success,
										session:cookies
									}), {
										headers:headers
									})
								}
							}
						}

					}catch(err){
						console.error('처리 중 오류 발생:', err)
					}
				}else{
					
				}

			}else if(request.method == 'GET'){
				if(contentType == 'application/json'){
					headers.set('Content-Type', contentType)

					var from = req.query.from

					if(ethers.isAddress(req.query.from)){
						from = req.query.from
					}

					var to = ""

					if(ethers.isAddress(req.query.to)){
						to = req.query.to
					}

					var ref = ""

					if(req.query.ref){
						if(ethers.isAddress(req.query.ref)){
							ref = req.query.ref
						}
					}

					var day = !isNaN(req.query.day) ? parseInt(req.query.day) : 0

					var condition = `AND "created_at" < ${created_at}`

					var conditions = {}


					var url = new URL(cookies.href)

					var link = (url.pathname+url.search).toLowerCase()



					var talks = []

					if(member && cookies.address){
						// relate update item
						var { results, success, error } = await env[`commerce_logis_${zoneRegion}_items`].prepare(
							`SELECT * FROM items WHERE "cc" = '${cookies.cc}' AND "to" = '${cookies.team}' ${condition} AND "updated_at" > 0 ORDER BY created_at DESC LIMIT 100`
						).all()

						console.log('results.length',results.length);

						if(results.length){
							for(var i = 0; i < results.length; i++){
								var item = results[i]

								item.table = 'items'

								if(Object.keys(detail).length){
									if(detail.id == item.id){
										continue
									}
								}

								items.push(item)
							}
						}


						// recent update item
						var { results, success, error } = await env[`commerce_logis_${zoneRegion}_items`].prepare(
							`SELECT * FROM items WHERE "cc" != '${cookies.cc}' AND "to" = '${cookies.team}' ${condition}  AND "updated_at" > 0 ORDER BY created_at DESC LIMIT 100`
						).all()

						if(results.length){
							for(var i = 0; i < results.length; i++){
								var item = results[i]

								item.table = 'items'

								if(Object.keys(detail).length){
									if(detail.id == item.id){
										continue
									}
								}

								items.push(item)
							}
						}


						// recent draft item
						var { results, success, error } = await env[`commerce_logis_${zoneRegion}_items`].prepare(
							`SELECT * FROM items WHERE "to" = '${cookies.team}' ${condition} AND "updated_at" = 0 ORDER BY created_at DESC LIMIT 100`
						).all()

						if(results.length){
							for(var i = 0; i < results.length; i++){
								var draft = results[i]

								draft.table = 'draft'

								cookies.draft[draft.ref]++

								if(Object.keys(detail).length){
									if(detail.id == draft.id){
										continue
									}
								}

								items.push(draft)
							}
						}


						var { results, success, error } = await env[logisRegion].prepare(
							`SELECT * FROM tasks WHERE "from" = '${cookies.address}' AND "cc" = '${cookies.cc}' ${condition} ORDER BY created_at DESC LIMIT 10`
						).all()

						if(results.length){
							for(var i = 0; i < results.length; i++){
								var task = results[i]

								task.table = 'tasks'

								task.ref = ''
								
								tasks.push(task)
							}
						}


						console.log('[GET] type',type);

						// `SELECT * FROM talks WHERE "cc" = '${cookies.cc}' AND "to" = '${pageId}' AND "data" IS NOT NULL AND "ref" = '${itemId}' ${conditions[page.type] ? conditions[page.type] : condition} ORDER BY created_at DESC LIMIT 10`

						var { results, success, error } = await env[`commerce_logis_${zoneRegion}_talks`].prepare(
							`SELECT * FROM talks WHERE "cc" = '${cookies.cc}' AND "data" IS NOT NULL AND "ref" = '${itemId}' ${conditions[page.type] ? conditions[page.type] : condition} ORDER BY created_at DESC LIMIT 10`
						).all()


						if(results.length){
							for(var i = 0; i < results.length; i++){
								var talk = results[i]
								talk.table = 'talks'

								var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(talk.data))

								var talk_data = JSON.parse(decompressedJsonString)
							
								if(talk_data){
									if(talk_data.search){
										delete talk_data.search	
									}
								}

								talk.data = gzip(new TextEncoder('utf-8').encode(JSON.stringify(talk_data)), { to: 'arraybuffer' })

								talks.push(talk)
							}

							var { results, success, error } = await env[`commerce_logis_${zoneRegion}_talks`].prepare(
								`SELECT COUNT(*) AS "talks" FROM talks WHERE "cc" = '${cookies.cc}' AND "data" IS NOT NULL AND "ref" = '${itemId}'`
							).all();

							if(results.length){
								var item = results[0]

								for (const itemType in item) {
									if (item.hasOwnProperty(itemType)) {
										cookies[itemType] = item[itemType]
									}
								}
							}
						}

						var { results, success, error } = await env[`commerce_logis_${zoneRegion}_talks`].prepare(
							`SELECT * FROM talks WHERE "from" = '${cookies.team}' AND "to" = '${cookies.address}' ${conditions[page.type] ? conditions[page.type] : condition} ORDER BY created_at DESC LIMIT 10`
						).all()

						if(results.length){
							for(var i = 0; i < results.length; i++){
								var talk = results[i]
								talk.table = 'talks'

								talks.push(talk)
							}
						}


						if(talks.length){
							talks = talks.sort((a, b) => b.created_at - a.created_at);
						}


						console.log('type',type);
						console.log('Object.keys(page).length',Object.keys(page).length);


						if(type){
							console.log(`SELECT * FROM ${type} WHERE "to" = '${cookies.team}' AND "cc" = '${cookies.cc}' AND "bcc" = '${page.bcc}' ${conditions[page.type] ? conditions[page.type] : condition} ORDER BY created_at DESC LIMIT 100`)

							if(Object.keys(detail).length){

								var { results, success, error } = await env[`commerce_logis_${zoneRegion}_items`].prepare(
									`SELECT * FROM items WHERE "type" = '${page.type}' AND "digest" = '${detail.digest}' ${condition} ORDER BY created_at DESC LIMIT 1`
								).all()

								if(results.length){
									var hints_condition = ' OR '

									for(var i = 0; i < results.length; i++){
										var hint = results[i]

										hints_condition += ` ("type" = '${page.type}' AND "id" = '${hint.id}' ${condition}) `
									}

									hints_condition = hints_condition.replace(' OR ', '')

									var { results, success, error } = await env[`commerce_logis_${zoneRegion}_${type}`].prepare(
										`SELECT * FROM ${type} WHERE ${hints_condition} ${condition} ORDER BY created_at DESC LIMIT 1`
									).all()


									if(results.length){
										var hints_condition = ' OR '

										for(var i = 0; i < results.length; i++){
											var hint = results[i]

											hint.table = 'hints'

											hints.push(hint)
										}
									}
								}

								console.log(`SELECT * FROM ${type} WHERE "id" = '${detail.id}' ${condition} ORDER BY created_at DESC LIMIT 1`);
								var { results, success, error } = await env[`commerce_logis_${zoneRegion}_${type}`].prepare(
									`SELECT * FROM ${type} WHERE "id" = '${detail.id}' ${condition} ORDER BY created_at DESC LIMIT 1`
								).all()

								console.log('detail item results.length',results.length);

								if(results.length){
									var $item = results[0]

									$item.table = type

									items.push($item)

									if(detail.type == "goods"){
										var { results, success, error } = await env[`commerce_logis_${zoneRegion}_sales`].prepare(
											`SELECT * FROM sales WHERE "goods" = ${$item.index} AND "cc" = '${cookies.cc}' AND "to" = '${cookies.team}' ${conditions[type] ? conditions[type] : condition} ORDER BY created_at DESC LIMIT 10000`
										).all()

										if(results.length){
											for(var i = 0; i < results.length; i++){
												var item = results[i]

												item.table = 'sales'

												if(Object.keys(detail).length){
													if(detail.id == item.id){
														continue
													}
												}

												items.push(item)
											}
										}

									}else if(detail.type == "order"){
										if($item.tracking){
											var { results, success, error } = await env[`commerce_logis_${zoneRegion}_tracking`].prepare(
												`SELECT * FROM tracking WHERE "order" = ${$item.index} AND "to" = '${cookies.team}' ${conditions[type] ? conditions[type] : condition} ORDER BY created_at DESC LIMIT 10000`
											).all()

											if(results.length){
												for(var i = 0; i < results.length; i++){
													var item = results[i]

													item.table = 'tracking'

													if(Object.keys(detail).length){
														if(detail.id == item.id){
															continue
														}
													}

													items.push(item)
												}
											}

										}else{
											var { results, success, error } = await env[`commerce_logis_${zoneRegion}_tracking`].prepare(
												`SELECT * FROM tracking WHERE "order" = ${item.index} AND "to" = '${cookies.team}' ${conditions[type] ? conditions[type] : condition} ORDER BY created_at DESC LIMIT 10000`
											).all()

											if(results.length){
												for(var i = 0; i < results.length; i++){
													var item = results[i]

													item.table = 'tracking'

													if(Object.keys(detail).length){
														if(detail.id == item.id){
															continue
														}
													}

													items.push(item)
												}
											}
										}
									}
								}

									
							}else{
								var { results, success, error } = await env[`commerce_logis_${zoneRegion}_${type}`].prepare(
									`SELECT * FROM ${type} WHERE "to" = '${cookies.team}' AND "cc" = '${cookies.cc}' AND "bcc" = '${page.bcc}' ${conditions[page.type] ? conditions[page.type] : condition} ORDER BY created_at DESC LIMIT 100`
								).all()

								
								console.log('{type} results.length',results.length);

								if(results.length){
									for(var i = 0; i < results.length; i++){
										var item = results[i]
										item.table = type

										if(Object.keys(detail).length){
											if(detail.id == item.id){
												continue
											}
										}

										items.push(item)
									}
								}


								console.log('page.type',page.type);
								console.log('type',type);

								console.log('zoneRegion',zoneRegion);

								// // list
								// if(page.type == "goods"){
								// 	var { results, success, error } = await env[`commerce_logis_${zoneRegion}_sales`].prepare(
								// 		`SELECT * FROM sales WHERE "type" = 'order' AND "cc" = '${cookies.cc}' AND "to" = '${cookies.team}' ${conditions[page.type] ? conditions[page.type] : condition} ORDER BY created_at DESC LIMIT 100`
								// 	).all()

								// 	console.log('list results.length',results.length)

								// 	if(results.length){
								// 		for(var i = 0; i < results.length; i++){
								// 			var item = results[i]

								// 			item.table = 'sales'

								// 			if(Object.keys(detail).length){
								// 				if(detail.id == item.id){
								// 					continue
								// 				}
								// 			}

								// 			items.push(item)
								// 		}
								// 	}

								// }else if(page.type == "order"){
								// 	var { results, success, error } = await env[`commerce_logis_${zoneRegion}_tracking`].prepare(
								// 		`SELECT * FROM tracking WHERE "cc" = '${cookies.cc}' AND "to" = '${cookies.team}' ${conditions[page.type] ? conditions[page.type] : condition} ORDER BY created_at DESC LIMIT 100`
								// 	).all()

								// 	console.log('list results.length',results.length)

								// 	if(results.length){
								// 		for(var i = 0; i < results.length; i++){
								// 			var item = results[i]

								// 			item.table = 'tracking'

								// 			if(Object.keys(detail).length){
								// 				if(detail.id == item.id){
								// 					continue
								// 				}
								// 			}

								// 			items.push(item)
								// 		}
								// 	}

								// }
							}

						}

						
					}


					return new Response(JSON.stringify({
						results:[...pages, ...tasks, ...talks, ...items, ...users, ...hints],
						session:cookies
					}), {
						headers:headers
					})
				}else{
					headers.set('Content-Type', 'text/html; charset=utf-8')

					return new Response(renderHtml(cookies ,[]), {
						headers:headers
					})
				}
			}else if(request.method == 'OPTIONS'){
				return new Response('', {
					headers:headers
				})
			}
		}catch(err){
			console.log('err',err)
		}

		return new Response(`I'm a teapot!`, {
			status:418,
			headers: { "Content-Type": "text/html; charset=utf-8" },
		})
	},
} satisfies ExportedHandler<Env>