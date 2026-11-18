(async function(){
	/*
		https://www.npmjs.com/package/js-obfuscator

		난독화 해서 복호화 막아보기

		AI 버튼 클릭을 한 영역에서 item selector 클릭 이벤트를 먼저 수집하여 

		item 상세페이지에서 이전 referrer 주소와 매칭하여 리스트와 아이템간의 데이터 구조화하기


		리스트 페이지
			window.location.href

		상세페이지
			document.referrer


		(카트 아이콘) Draft 2가지 타입
			고객 주문 등록(웹페이지 스캔)
			업체 재고 등록(택배송장 스캔)

	*/


	try{
		const app = {
			storage : {
				set : function(items) {
					return new Promise((resolve, reject) => {
						resolve()
					})
				},
				get : function(key) {
					return new Promise((resolve, reject) => {
						resolve(result)
					})
				},
				clear : function(){
					return new Promise((resolve, reject) => {
						resolve()
					})
				}
			},
			fetch : async function({ url, method = "GET", headers = {}, body = null }) {
				return new Promise((resolve, reject) => {
					resolve(response.json);
				});
			}
		}


		const Sleep = function(ms) {
			return new Promise(resolve => setTimeout(resolve, ms))
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

		var footprint = new URL(window.location.href)

		var { cookies } = await app.storage.get('cookies')

		if(!cookies?.hash){
			var { results, session } = await app.fetch({
				url : `https://logis.center?href=${encodeURIComponent(footprint.href)}`,
				method: "GET",
				headers: {
					"Content-Type": "application/json"
				}
			})

			cookies = session

			try{
				await app.storage.set({'cookies' : cookies})
				// await app.storage.set({'cookies' : {}})
			}catch(err){
				console.log('err',err);
			}
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




		var pathname = footprint.pathname

		var host = footprint.host
		
		var href = footprint.href

		var flag = Intl.DateTimeFormat().resolvedOptions().locale

		var isClient = isShop(href, clients)

		var isAdmin = isShop(href, admins)



		var timezoneOffset = new Date().getTimezoneOffset() * 60 * 1000

		var current = new Date(new Date().getTime() - timezoneOffset).getTime()

		var $current = new Date(current).toISOString()

		var selector = {
			dim : randomHash(),
			mcp : randomHash(),

			app : randomHash(),
			area : randomHash(),

			toggle : randomHash(),

			qrcode : randomHash(),
			qrauth : randomHash(),

			prompt : randomHash(),
			context : randomHash(),
			submit :  randomHash(),
			result : randomHash(),
			results : randomHash(),
			scroll : randomHash(),

			hidden : randomHash(),
			$list : randomHash(),
			syntax : randomHash(), // 동기화 전
			$yntax : randomHash(), // 동기화 후
			visited : randomHash() // 상세 동기화 후
		}

		var isLock = false

		var isFocus = false

		window.addEventListener("blur", async function(event) {
			isFocus = false

			try{
				if(window[cookies.hash]){
					timeout.clear()
				}
			}catch(err){

			}
		})

		window.addEventListener("focus", async function(event) {
			isFocus = true

			try{
				var { cookies } = await app.storage.get('cookies')

				if(!window[cookies.hash]){
					window[cookies.hash] = setTimeout(timeout.fn, timeout.ms)
				}
			}catch(err){

			}
		})


		const removeChild = (d) => d && d.parentNode && d.parentNode.removeChild(d)




		

		var retryCount = 0

		var $qrcode

		var $qrauth

		var onAuth = async function(e){
			timeout.ms = 1000
			retryCount = 1

			$qrauth.textContent = "Loading"

			var { cookies } = await app.storage.get('cookies')
			
			window[cookies.hash] = setTimeout(timeout.fn, timeout.ms)
		}

		var $app = document.createElement("div")
			$app.className = selector.app

		const placeholder = {
			prompt : function(){
				return cookies.address ? "Enter prompt" : "Please scan the QR code to verify"
			},
			confirm : "문서를 동기하시겠습니까?"
		}

		var $Style = document.createElement("style")
			$Style.innerHTML = `
				[class="${selector.app}"],
				[class="${selector.area}"] *,
				[class="${selector.area}"] *::selection{all: initial !important; background-color: transparent;}

				[class="${selector.app}"]{position: fixed !important; right: 0 !important; top: 0 !important; bottom: 0 !important; max-width: 360px !important; width: 100% !important; pointer-events: none !important; z-index:100000000000000 !important;}

				[class="${selector.area}"]{position: absolute !important; right: 0 !important; bottom: 50px !important; max-width: 100% !important; width: 100% !important; max-width: 340px !important; pointer-events:none !important;}

				[class="${selector.qrcode}"]{display:none !important; margin:10px auto 3px !important; padding: 10px !important; border-radius: 10px !important; max-width: 300px !important; border-radius: 10px !important; background-color: #fff !important; box-shadow: 0 0 20px 0px #00000055 !important;}

				[class="${selector.qrcode}"] canvas{display:none !important;}

				[class="${selector.qrcode}"] img{display:block !important; width:100% !important; max-width:100% !important; min-width: 100% !important;}

				[class="${selector.qrauth}"]{display: none !important; margin: 15px 10px !important; border-radius: 0.7em !important; height: 3em !important; line-height: 3em !important; background-color: #000 !important; color: #fff !important; font-weight: 900 !important; text-transform: uppercase !important; text-decoration: underline !important; text-align: center !important; cursor: pointer !important; box-shadow: 0 0 20px 0px #00000055 !important;}


				[class="${selector.prompt}"]{position: relative !important; display: none !important; width: 100% !important; padding: 10px !important; box-sizing: border-box !important;}

				[name="${selector.prompt}"]{overflow: hidden !important; position: relative !important; display: block !important; width: 100% !important; height: 125px !important; border-radius: 10px !important; background-color: #fff !important; box-shadow: 0 0 20px 0px #00000055 !important;}

				[name="${selector.context}"]{overflow: hidden !important; overflow-y: scroll !important; display: block !important; margin-bottom:45px !important; padding: 10px 14px 0 !important; width: 100% !important; height: 80px !important; line-height: 1.2 !important; font-size: 15px !important; white-space: pre-wrap !important; overflow-wrap: break-word !important; box-sizing: border-box !important;}

				[for="${selector.submit}"]{overflow: hidden !important; position: absolute !important; right: 10px !important; bottom: 10px !important; border-radius: 50% !important; pointer-events:initial !important; cursor:pointer !important; z-index: 1 !important;}

				[for="${selector.submit}"] img{display:block !important; width: 25px !important; height: 25px !important; cursor:pointer !important;}

				[id="${selector.submit}"]{display:none !important; pointer-events:initial !important;}


				[class="${selector.mcp}"]{position: absolute !important; left: 10px !important; bottom: 10px !important; padding: 0 8px 0 5px !important; border-radius: 10px !important; height: 25px !important; line-height: 24px !important; font-size: 12px !important; text-align: center !important; pointer-events: initial !important; cursor: pointer !important; background: #000000 !important; color: #fff !important;}
				[class="${selector.mcp}"]::selection{color: #fff !important;}

				[class="${selector.hidden}"]{position: absolute !important; right: 0 !important; bottom:0 !important; width:100% !important; height: 100% !important; z-index: -1 !important;}

				[for="${selector.toggle}"]{position: absolute !important; right: 10px !important; bottom: -38px !important; margin: 0 auto !important; border-radius: 12px !important; width: 36px !important; height: 36px !important; text-align: center !important; font-size: 25px !important; font-family: 'Noto Color Emoji' !important; box-shadow: 0 0 4px #000 !important; background: #000000d6 !important; cursor: pointer !important; z-index: 100 !important; pointer-events: initial !important; transform:scale(1.3) !important; transition-duration: 0.2s !important;}
				[for="${selector.toggle}"]:after{content:"✨"; display: block !important; text-indent: 1.5px !important; line-height: 32.5px !important; font-size:15px !important; transition-duration: 0.2s !important;}


				[class*="${selector.syntax}"]{opacity:0.5 !important;}
				[class*="${selector.$yntax}"] *{font-weight:100 !important;}


				[class*="${selector.visited}"],
				[class*="${selector.visited}"] *{font-weight:bold !important;}

				[class*="${selector.syntax}"],
				[class*="${selector.$yntax}"]{position:relative !important; z-index: 0 !important;}


				[id="${selector.toggle}"]{display: none !important; pointer-events: initial !important;}


				[id="${selector.toggle}"]:checked+[class="${selector.area}"] [class="${selector.qrauth}"],
				[id="${selector.toggle}"]:checked+[class="${selector.area}"] [class="${selector.qrcode}"],
				[id="${selector.toggle}"]:checked+[class="${selector.area}"] [class="${selector.prompt}"]{display:block !important;}

				[id="${selector.toggle}"]:checked+[class="${selector.area}"] [class="${selector.qrauth}"]+[class="${selector.prompt}"]{display:none !important;}

				[id="${selector.toggle}"]:checked+[class="${selector.area}"] [for="${selector.toggle}"]{transform:scale(1) !important;}

				[id="${selector.toggle}"]:checked+[class="${selector.area}"] [for="${selector.toggle}"]:after{content:"❌";}


				.animated-background {
					animation-duration: 2s;
					animation-fill-mode: forwards;
					animation-iteration-count: infinite;
					animation-name: placeHolderShimmer;
					animation-timing-function: linear;
					background-color: #f6f7f8;
					background: linear-gradient(to right, #eeeeee 8%, #bbbbbb 18%, #eeeeee 33%);
					background-size: 800px 104px;
					height: 70px;
					position: relative;
				}

				@keyframes placeHolderShimmer {
					0% {
						background-position: -800px 0;
					}
					100% {
						background-position: 800px 0;
					}
				}

			`


		var logo = `<img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZlcnNpb249IjEuMiIgdmlld0JveD0iMC45NTQ3Mzg2MTY5NDMzNTk0IDAuOTU0NzQyNDMxNjQwNjI1IDc4LjA5MDUxNTEzNjcxODc1IDc4LjA5MDUxNTEzNjcxODc1IiB3aWR0aD0iNTAwIiBoZWlnaHQ9IjUwMCI+Cgk8dGl0bGU+TDhiVVZqdENyR0NHWktSeDBFejU4Qm5nYklEekcyeHh1WHZRcFRmcHNzV0hDSkFJRDVVeGVfN3Q0Zi12NlBhSS1IZ2FkU0JqYkJMTVN6WFBfVHlkMnc8L3RpdGxlPgoJPHN0eWxlPgoJCS5zMCB7IGZpbGw6ICMwMDAwMDAgfSAKCQkuczEgeyBmaWxsOiAjZmZmZmZmIH0gCgk8L3N0eWxlPgoJPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGFzcz0iczAiIGQ9Im0yLjMgMjkuOWM1LjYtMjAuOCAyNy0zMy4yIDQ3LjgtMjcuNiAyMC44IDUuNiAzMy4yIDI3IDI3LjYgNDcuOC01LjYgMjAuOC0yNyAzMy4yLTQ3LjggMjcuNi0yMC44LTUuNi0zMy4yLTI3LTI3LjYtNDcuOHoiLz4KCTxwYXRoIGNsYXNzPSJzMSIgZD0ibTc2LjcgNDkuOGMyLjYtOS43IDEuMi0yMC4xLTMuOC0yOC44LTUtOC43LTEzLjMtMTUuMS0yMy4xLTE3LjctOS43LTIuNi0yMC4xLTEuMi0yOC44IDMuOC04LjcgNS0xNS4xIDEzLjMtMTcuNyAyMy4xLTEuMyA0LjgtMC42IDEwIDEuOSAxNC40IDIuNSA0LjMgNi43IDcuNSAxMS41IDguOCA0LjkgMS4zIDEwLjEgMC43IDE0LjQtMS45IDQuNC0yLjUgNy42LTYuNiA4LjktMTEuNSAxLjMtNC45IDQuNS05IDguOS0xMS41IDQuMy0yLjYgOS41LTMuMiAxNC40LTEuOSA0LjggMS4zIDkgNC41IDExLjUgOC44IDIuNSA0LjQgMy4yIDkuNiAxLjkgMTQuNHoiLz4KPC9zdmc+" />`

		document.head.appendChild($Style)

		var formTpl = function(){
			return `
				<div class="${selector.prompt}">
					<div class="${selector.results}">
						<div class="${selector.scroll}">
						</div>
					</div>

					<form name="${selector.prompt}">
						<textarea ${cookies.address ? "" : "disabled"} maxlength="140" cols="40" rows="2" name="${selector.context}" placeholder="${placeholder.prompt()}"></textarea>

						<label for="${selector.submit}">
							${logo}
							<input id="${selector.submit}" type="submit">
						</label>

						<a class="${selector.mcp}">✨${isMobile ? "QR Scan" : "List Scan"}</a>
					</form>
				</div>
				<label for="${selector.toggle}"></label>
			`
		}

		console.log('cookies',cookies)

		document.body.appendChild($app)


		var $results = new MtfScrollList()

		$results.init({
			ele: document.querySelector(`[class="${selector.scroll}"]`),
			data: [],
			startIndex: 0,
			perPage: 0,
			render ({ item, index }) {
				const d = document.createElement('div')
				d.setAttribute('index', index)
				d.id = 'id' + index

				if(item.data){
					const decompressedJsonString = new TextDecoder('utf-8').decode(pako.ungzip(item.data))

					item.data = JSON.parse(decompressedJsonString)
				}

				var status = ""

				body += `<label for="_${item.id}" class="${selector.result}">${status} ${item.type} </label>`

				body += `<input type="checkbox" id="${item.id}" />`

				body += `<div id="${item.id}" class="${selector.result}">`

				var created_at = new Date(new Date().getTime() - timezoneOffset).toISOString()

				if(item.type == "sales"){
					if(item.title){
						body += `<div>${item.title}</div>`
					}

					if(item.width){
						body += `<div>${item.width} cm</div>`
					}

					if(item.height){
						body += `<div>${item.height} cm</div>`
					}

					if(item.length){
						body += `<div>${item.length} cm</div>`
					}

					if(item.weight){
						body += `<div>${item.weight} kg</div>`
					}

					if(item.price){
						body += `<div>${item.price} ${item.currency}</div>`
					}

					if(item.cost_price){
						body += `<div>${item.cost_price} ${item.currency}</div>`
					}

					if(item.sale_price){
						body += `<div>${item.sale_price} ${item.currency}</div>`
					}

					if(item.discount){
						body += `<div>${item.discount} ${item.currency}</div>`
					}

					if(item.quantity){
						body += `<div>${item.quantity}</div>`
					}

					if(item.reward_point){
						body += `<div>${item.reward_point}</div>`
					}

					if(item.shipping_fee){
						body += `<div>${item.shipping_fee} ${item.currency}</div>`
					}

					if(item.shipping_method){
						body += `<div>${item.shipping_method}</div>`
					}

					if(item.shipping_duration){
						body += `<div>${item.shipping_duration}</div>`
					}

					if(item.tax_included){
						body += `<div>${item.tax_included}</div>`
					}

					if(item.release_date){
						body += `<div>${item.release_date}</div>`
					}

					if(item.manufacture_date){
						body += `<div>${item.manufacture_date}</div>`
					}

					if(item.expiration_date){
						body += `<div>${item.expiration_date}</div>`
					}

					if(item.status){
						body += `<div>${item.status}</div>`
					}

				}else if(item.type == "track"){
					if(item.status){
						body += `<div>${item.status}</div>`
					}

					if(item.data){
						if(item.data.senderName){
							body += `<div>${item.data.senderName}</div>`
						}

						if(item.data.senderAddress){
							body += `<div>${item.data.senderAddress}</div>`
						}

						if(item.data.senderPhone){
							body += `<div>${item.data.senderPhone}</div>`
						}


						if(item.data.recipientName){
							body += `<div>${item.data.recipientName}</div>`
						}

						if(item.data.recipientAddress){
							body += `<div>${item.data.recipientAddress}</div>`
						}

						if(item.data.recipientPhone){
							body += `<div>${item.data.recipientPhone}</div>`
						}
					}

				}else if(item.type == "event"){
					if(item.status){
						body += item.status
					}

					if(item.title){
						body += item.title
					}

					if(item.code){
						body += item.code
					}

					if(item.discount){
						body += item.discount
					}

					if(item.quantity){
						body += item.quantity
					}

					if(item.usage_per){
						body += item.usage_per
					}

					if(item.usage_limit){
						body += item.usage_limit
					}

					if(item.new_customer_only){
						body += item.new_customer_only
					}

					if(item.min_order_amount){
						body += item.min_order_amount
					}

					if(item.max_discount_amount){
						body += item.max_discount_amount
					}

					if(item.first_purchase_only){
						body += item.first_purchase_only
					}

					if(item.region_restrictions){
						body += item.region_restrictions
					}
				}


				body += `<input type="date" readonly name="created_at" value="${new Date(created_at)}" />`

				body += `</div>`


				try{
					if(typeof item.job != "undefined"){
						await Upsert['crons'](item)

						// render virtualized-list

					}else{
						await Upsert["items"](item);
					}
				}catch(err){

				}
				// d.innerHTML = data.text
				return d
			},
			onTop ({ cb, page }) {
				setTimeout(() => {
					cb(data)
				}, 1500)
			},
			onBottom ({ cb, page }) {
				setTimeout(() => {
					cb(data)
				}, 0)
			},
			onPullDownStart ({ startY }) {
				let d = document.querySelector(`[class="${selector.pullover}"]`)

				removeChild(d)
				
				d = document.createElement('div')
				d.className = selector.pullover

				document.body.appendChild(d)
			},
			onPullDownMove ({ paddingTop }) {
				const d = document.getElementById(`[class="${selector.pullover}"]`)

				if(paddingTop > 100){
					return true
				}

				d.style.marginTop = (paddingTop >> 1) + 'px'
			},
			onPullDownEnd ({ paddingTop, cb }) {
				const d = document.getElementById(`[class="${selector.pullover}"]`)

				if (paddingTop >= 50) {
					setTimeout(() => {
						removeChild(d)
						cb(data)
					}, 1500)
				} else {
					removeChild(d)
				}
			}
		})

		var timeout = {
			ms : 5000,
			clear : async function(){
				var { cookies } = await app.storage.get('cookies')

				clearTimeout(window[cookies.hash])
				window[cookies.hash] = null
			},
			fn : async function(init){
				timeout.clear()

				try{
					var { cookies } = await app.storage.get('cookies')

					var params = `&referrer=${document.referrer}`

					if(document.referrer && cookies.team){
						var url = new URL(document.referrer)

						var ref = hashId(cookies.team+cookies.cc+url.pathname+url.search)

						var id = hashId(ref)

						var pageId = hashId(cookies.cc+url.pathname+url.search)


						try{
							var items = await Select['items']({
								id : ref
							})

							var crons = await Select['crons']({
								id : ref
							})

							if(crons.length){


							}else{
								var body = pako.gzip(new TextEncoder('utf-8').encode(document.body.innerHTML), { to: 'arraybuffer' })

								var created_at = new Date(new Date().getTime() - timezoneOffset).getTime()

								var { results } = await app.fetch({
									url : `https://logis.center?hash=${cookies.hash}&token=${cookies.token}&from=${cookies.address}&to=${cookies.team}&created_at=${created_at}&href=${encodeURIComponent(window.location.href)}`,
									method: 'POST',
									headers: {
										'Content-Type': 'application/octet-stream',
										'Content-Encoding': 'gzip'
									},
									body : body.buffer
								})

								await Upsert['crons'](results[0]);
							}


							if(items.length){
								// 렌더링
								if(!$results){
									$results = VirtualizedList(items)
								}else{
									// append 해야함
									$results.dom.append(items)
								}
							}

							// if(pages.length){
							// 	var page = pages[0]
							// }
						}catch(err){
							console.log('err',err);
						}
					}


					var { results, session } = await app.fetch({
						url : `https://logis.center?hash=${cookies.hash}&token=${cookies.token}&created_at=${current}&ref=${Ethers.ZeroAddress}&href=${encodeURIComponent(window.location.href)}${params}`,
						method: "GET",
						headers: {
							"Content-Type": "application/json"
						}
					})

					console.log('session',session);


					if($qrauth){
						if(cookies.address){
							retryCount = 0
							
							timeout.ms = 5000

							var $context = $app.querySelector(`[name="${selector.context}"]`)

							$context.setAttribute('placeholder', placeholder.prompt())
							$context.removeAttribute('disabled')

							$qrauth.removeEventListener("click", onAuth)
							$qrauth.remove()
							$qrcode.remove()

							$qrcode = null
							$qrauth = null
							
						}else if(retryCount){
							var dots = ""

							for(var d = 0; d < retryCount; d++){
								dots += "."
							}

							retryCount += 1

							if(retryCount > 3){
								retryCount = 1
							}

							$qrauth.textContent = `Loading${dots}`	
						}
					}

					try{
						console.log('session',session);
						await app.storage.set({'cookies' : session})

						cookies = session
						// await app.storage.set({'cookies' : {}})
					}catch(err){
						console.log('err',err);
					}

					console.log('results',results);

					var ref = hashId(session.team+session.cc+footprint.pathname+footprint.search)

					var id = hashId(ref)

					var pageId = hashId(cookies.cc+footprint.pathname+footprint.search)


					// var items = await Select['items']({
					// 	id : ref
					// })

					if(session.address){
						if(cookies.address){
							if(session.address != cookies.address){
								// 세션 "미일치"시 제거
								// await Clear['items']()
								// await Clear['pages']()
								// await Clear['crons']()

							}
						}

						var items = []

						var crons = []

						if(results?.length){
							// crons 값 동기화
							for(var i = 0; i < results.length; i++){
								var item = results[i]

								if(item.job){
									crons.push(item)

									try{
										await Upsert['crons'](item)
									}catch(err){

									}

								}else if(typeof item.vector == "undefined"){
									try{
										if(item.data){
											const decompressedJsonString = new TextDecoder('utf-8').decode(pako.ungzip(item.data))

											var obj = JSON.parse(decompressedJsonString)

											for (const key in obj) {
												if (obj.hasOwnProperty(key)) {
													selector[key] = obj[key]
												}
											}

											try{
												await Upsert['pages'](item);
											}catch(err){

											}
										}
									}catch(err){
										console.log('page err',err);
									}

								}else{
									items.push(item)

									try{
										await Upsert['items'](item);
									}catch(err){

									}
								}
							}



							if(selector.item){
								// 프롬프트 결과 아이템
								if(items.length){
									for(var i = 0; i < items.length; i++){
										var item = items[i]

										
									}
								}


								// 내용 여부 없으면 결과값 찾기

								var $items = document.querySelectorAll(selector.item)

								console.log('$items',$items);

								var list = []


								if($items.length){
									for(var i = 0; i < $items.length; i++){
										var $item = $items[i]

										var $list = $item.querySelectorAll('a')

										if($list.length){
											for(var s = 0; s < $list.length; s++){
												var $link = $list[s]

												$link.setAttribute('target', "_blank")

												$link.removeAttribute('rel')
												$link.removeAttribute('referrerpolicy')

												// var $footprint = document.createElement('div');
											}
										}


										var $mores = $item.querySelectorAll(selector.more)

										if($mores.length){
											if($mores.length){
												for(var m = 0; m < $mores.length; m++){
													var $more = $mores[m]

													try{
														var link = new URL($more.href)

														list.push(link.href)

														break;
													}catch(err){
														console.log('err',err);
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

									console.log('selector',selector)

									if(list.length){
										var created_at = new Date(new Date().getTime() - timezoneOffset).getTime()

										var formData = new FormData()

										formData.append('list', list)

										var { results, session } = await app.fetch({
											url : `https://logis.center?hash=${cookies.hash}&token=${cookies.token}&href=${encodeURIComponent(window.location.href)}&created_at=${created_at}`,
											method: "POST",
											body : formData
										})

										var ref = hashId(session.team+session.cc+footprint.pathname+footprint.search)

										var id = hashId(ref)


										// var { results, session } = await app.fetch(`https://logis.center?hash=${cookies.hash}&token=${cookies.token}&href=${encodeURIComponent(window.location.href)}&list=${list.toString()}&created_at=${created_at}`, {
										// 	method: "GET",
										// 	headers: {
										// 		"Content-Type": "application/json"
										// 	}
										// })

										console.log('results',results)


										var syncItem = {}

										var sync = []

										var $list = document.querySelector(selector.list)

										console.log('selector.list',selector.list);


										var isScan = false

										var $links = $list.querySelector('a')


										var temp = {}

										for(var a = 0; a < $links.length; a++){
											var $link = $links[a]

											try{
												var _url = new URL($link.href)

												var ref = hashId(cookies.team+cookies.cc+_url.pathname+_url.search)

												var id = hashId(ref)

												console.log('id',id);
												console.log('ref',ref);

												temp[id] = $link.href

												temp[ref] = $link.href

											}catch(err){
												console.log('err',err, $link.href);
											}
										}

										for(var r = 0; r < results.length; r++){
											var item = results[r]

											try{
												var _ref = item.id

												var _id = item.ref

												var _link = temp[_ref] || temp[_id]

												if(_link){
													sync.push(_link)
												}

												if(_id == id && _ref == ref){
													// 동기화 완료 영역
													isScan = true

													$item.classList.add(selector.visited)
												}else{
													$item.classList.add(selector.$yntax)
												}

											}catch(err){
												console.log('err',err);
											}
										}


										console.log('sync',sync);
										console.log('list',list);

										// 클릭 안된 링크
										var hiddens = list.filter(link => !sync.includes(link));

										console.log('hiddens',hiddens);
										if(hiddens.length){
											for(var i = 0; i < hiddens.length; i++){
												var hidden = hiddens[i]

												try{
													var url = new URL(hidden)
													var link = url.pathname + url.search

													var $link = $list.querySelector(`[href="${link}"]`)

													var $item = $link.closest(selector.item)
														$item.classList.add(selector.syntax)

												}catch(err){
													console.log('err',err);
												} 
											}
										}
									}
								}
							}
						}
					}

					if(isFocus && session.address || retryCount){
						window[cookies.hash] = setTimeout(timeout.fn, timeout.ms)
					}
				}catch(err){
					console.log('err',err);
				}

				if(init){
					console.log('cookies',cookies);
				}
			}
		}

		if(isClient || isAdmin){

			if(isClient){ 
				// 재고 조회

				if(footprint){
					
				}else{
					
				}

			}else if(isAdmin){ 
				// 주문 조회	

				if(footprint){
					
				}else{
					
				}
			}

			if(cookies.address){
				$app.innerHTML = `<div class="${selector.dim}"></div><input type="checkbox" id="${selector.toggle}" />
				<div class="${selector.area}">
					${formTpl()}
				</div>`

				var $form = $app.querySelector(`form[name="${selector.prompt}"]`)

				$form.addEventListener('submit', async function(e){
					e.preventDefault()

					if($form[selector.context].value){
						var created_at = new Date(new Date().getTime() - timezoneOffset).getTime()

						var formData = new FormData()

						formData.append('text', $form[selector.context].value)

						var { cookies } = await app.storage.get('cookies')

						var { results } = await app.fetch({
							url : `https://logis.center?hash=${cookies.hash}&token=${cookies.token}&from=${cookies.team}&to=${cookies.address}&created_at=${created_at}&href=${encodeURIComponent(window.location.href)}`,
							method: 'POST',
							body : formData
						})

						console.log('results',results)


						// var body = "";

						// if(results?.length){
						// 	for(var i = 0; i < results.length; i++){
								
						// 	}
						// }
					}

					return
				})


				$form.querySelector(`[class="${selector.mcp}"]`).addEventListener('click', async function(e){
					try{
						var body = pako.gzip(new TextEncoder('utf-8').encode(document.body.innerHTML), { to: 'arraybuffer' })

						var created_at = new Date(new Date().getTime() - timezoneOffset).getTime()

						var href = window.location.href


						var { cookies } = await app.storage.get('cookies')

						var { results } = await app.fetch({
							url : `https://logis.center?hash=${cookies.hash}&token=${cookies.token}&from=${cookies.address}&to=${cookies.team}&created_at=${created_at}&href=${encodeURIComponent(href)}`,
							method: 'POST',
							headers: {
								'Content-Type': 'application/octet-stream',
								'Content-Encoding': 'gzip'
							},
							body : body.buffer
						})

						console.log('results',results)

						if(results?.length){
							var item = results[0]

							if(item.ref){
								// 결과 노출해야함
								// virtualized-list


							}else{
								try{
									await Upsert['crons']({
										id : item.id,
										cc : item.cc,
										bcc : item.bcc,
										ref : item.ref,
										job : body.buffer,
										created_at : item.created_at,
										updated_at : item.updated_at
									})
								}catch(err){

								}
							}
						}
					}catch(err){
						console.log('err',err);
					}

				})


				/*
					click event 이후에

					pages.item값이 hide 되면

					tab이벤트가 발생했으니 새로 scan 이벤트 실행해야함

					list selector에서 
				*/

				if(selector.list && selector.item){
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

								var $list = $this.closest(selector.list)

								console.log('selector.list',selector.list);

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

			}else{
				$app.innerHTML = `<input type="checkbox" id="${selector.toggle}" />
				<div class="${selector.area}">
					<div class="${selector.qrcode}"></div>
					<a class="${selector.qrauth}">QR Verify</a>

					${formTpl()}
				</div>
				`

				$qrauth = $app.querySelector(`[class="${selector.qrauth}"]`)

				$qrauth.addEventListener('click', onAuth)

				/*
					로그인하는 UI 필요함

					성공하면 cookies 값 받아와야함
				*/ 

				$qrcode = $app.querySelector(`[class="${selector.qrcode}"]`)

				new QRCode($qrcode, {
					text: "mailto:"+encodeURIComponent(cookies.hash+".logis.center@oauth.email"),
					width: 300,
					height: 300,
					colorDark : "#000000",
					colorLight : "#ffffff",
					correctLevel : QRCode.CorrectLevel.H
				})

			}


			var $toggle = $app.querySelector(`[id="${selector.toggle}"]`)

			$toggle.addEventListener('click', async function(){
				try{
					isFocus = true

					console.log('$toggle.checked',$toggle.checked);

					var { cookies } = await app.storage.get('cookies')

					if($toggle.checked){
						timeout.clear()

						window[cookies.hash] = setTimeout(timeout.fn, timeout.ms)
					}else{
						if(window[cookies.hash]){
							timeout.clear()
						}
					}
				}catch(err){
					console.log('err',err);
				}
			})
		}
	}catch(err){
		console.log('err',err);
	}

}())