; (function() {	
	window.crc32 = function(r){for(var a,o=[],c=0;c<256;c++){a=c;for(var f=0;f<8;f++)a=1&a?3988292384^a>>>1:a>>>1;o[c]=a}for(var n=-1,t=0;t<r.length;t++)n=n>>>8^o[255&(n^r.charCodeAt(t))];return(-1^n)>>>0};
	
	if (!Array.prototype.filter) {
		Array.prototype.filter = function(fun /*, thisArg*/ ) {
			'use strict'
			if (this === void 0 || this === null) {
				throw new TypeError()
			}
			var t = Object(this)
			var len = t.length >>> 0
			if (typeof fun !== 'function') {
				throw new TypeError()
			}
			var res = []
			var thisArg = arguments.length >= 2 ? arguments[1] : void 0
			for (var i = 0; i < len; i++) {
				if (i in t) {
					var val = t[i]

					if (fun.call(thisArg, val, i, t)) {
						res.push(val)
					}
				}
			}
			return res
		}
	}
	if (!Object.entries) Object.entries = function(obj) {
		var ownProps = Object.keys(obj),
			i = ownProps.length,
			resArray = new Array(i)
		while (i--) resArray[i] = [ownProps[i], obj[ownProps[i]]]
		return resArray
	}

	Array.prototype.findIndex = Array.prototype.findIndex || function(callback) {
		if (this === null) {
			throw new TypeError('Array.prototype.findIndex called on null or undefined')
		} else if (typeof callback !== 'function') {
			throw new TypeError('callback must be a function')
		}
		var list = Object(this)

		var length = list.length >>> 0
		var thisArg = arguments[1]
		for (var i = 0; i < length; i++) {
			if (callback.call(thisArg, list[i], i, list)) {
				return i
			}
		}
		return -1
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

	if (typeof Element !== "undefined") {
		if (!Element.prototype.matches) {
			Element.prototype.matches = Element.prototype.msMatchesSelector || Element.prototype.webkitMatchesSelector
		}

		if (!Element.prototype.closest) {
			Element.prototype.closest = function (s) {
				var el = this

				do {
					if (el.matches(s)) return el
					el = el.parentElement || el.parentNode
				} while (el !== null && el.nodeType === 1)
				
				return null
			}
		}
	}

	window.truncate = function( str, max, sep ) {
		max = max || 14;
		var len = str.length;
		if(len > max){
			sep = sep || "...";
			var seplen = sep.length;
			if(seplen > max) { 
				return str.substring(len - max)
			}
			var n = -0.5 * (max - len - seplen);
			var center = len/2;
			return str.substring(0, center - n) + sep + str.substring(len - center + n);
		}
		return str;
	}

	var OAuth3 = {}

		OAuth3.host = window.location.host.indexOf("localhost") > -1 ? "http://localhost:3000" : "https://cdn.oauth.network"

		OAuth3.localhost = window.location.host.indexOf("localhost") > -1 ? "popup.run.goorm.io" : ""

		OAuth3.rows = {}

		OAuth3.isMobile = ""

		OAuth3.on = function(type, listener){
			var t = typeof type == "object" ? type.type : type
			var l = typeof type == "object" ? type.listener : listener

			if(t == "load"){
				OAuth3.on[t] = l
				
			}else if(t == "ready"){
				window.addEventListener("DOMContentLoaded", function(e) {
					l(e)
				})
			}else{
				return window.addEventListener(t, function(e){
					e.OAuth3 = OAuth3
					e.form = e.target.closest('form')

					if(e.form){
						l(e)
					}
				})
			}
		}

		OAuth3.off = function(type, listener){
			var t = typeof type == "object" ? type.type : type
			var l = typeof type == "object" ? type.listener : listener

			if(t == "load"){
				OAuth3.on[t] = undefined
			}else{
				window.removeEventListener(t, l)
			}
		}

		if (navigator.userAgent) {
			if((/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent))){
				if((/iphone|ipad|ipod/i.test(navigator.userAgent.toLowerCase()))){
					OAuth3.isMobile = "ios"
				}else{
					OAuth3.isMobile = "aos"
				}
			}
		}
		
		OAuth3.network = []
		OAuth3.fetch = function(req, res){
			var tasksLength = OAuth3.network.length
			var state = crc32(JSON.stringify(req)+new Date().getTime().toString()).toString(32).toUpperCase()
			
			var params = ""
			
			var promise
			
			if(typeof req == "undefined" && typeof res == "undefined"){
				if(tasksLength){
					promise = true
					state = OAuth3.network[0]
					
					if(state){
						if(OAuth3.network[state]){
							req = OAuth3.network[state].request
							res = OAuth3.network[state].response
						}
					}
				}
			}
			
			if(req){
				var task = ""
				var method = req.method + ""
				
				if(typeof req.url == "undefined"){
					req.url = OAuth3.host+""
				}

				if(req.query){
					req.query.state = state
				}else{
					req.query = {
						state : state
					}
				}

				if(req.url.indexOf('amazonaws.com') > -1){
					delete req.query.state
				}

				if(req.query){
					params = "?"+new createSearchParams(req.query).toString()
					
					task = "["+method+"]"+req.url+params
				}
				
				if(req.mailto){
					task = "["+method+"]"+req.mailto
				}
				
				if(req.body){
					task = "["+method+"]"+req.url+params+JSON.stringify(req.body)
				}
				
				if(req.formData){
					task = "["+method+"]"+req.url+params+JSON.stringify(req.formData)
				}
				
				// if(method == "POST" && typeof OAuth3.session == "undefined"){
				// 	task = undefined
				// }

				var iframe = document.createElement("iframe")
					iframe.name = state

				if(task && !promise){
					if(req.count){
						var len = req.count

						for(var i = 0; i < len; i++){
							OAuth3.network.push(state)
						}
					}else{
						OAuth3.network.push(state)
					}
					
					OAuth3.network[state] = {
						request : req,
						response : res
					}
				}
				
				if(method == "GET" || method == "DELETE"){
					var $state = document.querySelector('head iframe[name="'+state+'"]');

					var count = 0
					
					for(var i = 0; i < OAuth3.network.length; i++){
						if(OAuth3.network[i] == state){
							count++
						}
					}
					
					if(req.mailto && req.count == count){
						OAuth3.network.shift()
						
						if(req.compose){
							var features = ""

							if(req.compose == "apple"){
								features = 'noopener'
							}

							OAuth3.opener = window.open(req.mailto, "oauth.email", features)
							OAuth3.fetch()
							
						}else{
							if(OAuth3.isMobile == "ios"){
								window.top.location.href = req.mailto
							}else{
								if(!$state){
									document.head.appendChild(iframe)
									iframe.src = req.mailto
								}
							}
						}
					}else{
						if(!$state){
							document.head.appendChild(iframe)
							iframe.src = req.url+params
						}
					}
				}else if(method == "POST"){
					var form = document.createElement("form")
						form.method = method
						form.target = iframe.name
						form.action = req.url+params

					try{
						if(req.body){
							form.enctype = "application/x-www-form-urlencoded"
							
							if(Object.keys(req.body).length){
								for(var name in req.body){
									if(req.body.hasOwnProperty(name)) {
										var values = req.body[name]
										
										if(typeof values == "object"){
											try{
												var len = values.length

												if(len){
													for(var r = 0; r < len; r++){
														var value = values[r]

														var $input = document.createElement("input")
															$input.name = name
															$input.type = "hidden"
															$input.value = JSON.stringify(value)

														form.appendChild($input)
													}
												}
											}catch(err){
												
											}	
										}else if(typeof values == "string"){
											var $input = document.createElement("input")
												$input.name = name
												$input.type = "hidden"
												$input.value = values

											form.appendChild($input)
										}
									}
								}
							}
						}else if(req.formData){
							form.enctype = "multipart/form-data"

							for(var name in req.formData){
								if(req.formData.hasOwnProperty(name)) {
									var input = document.createElement("input")
										input.type = "hidden"
										input.name = name
										input.value = req.formData[name]
									
									form.appendChild(input)
								}
							}

							if(req.blob){
								var file = new File([req.blob], req.blob.name, {type : req.blob.type, lastModified:req.blob.lastModified}, 'utf-8')
								var container = new DataTransfer() 
									container.items.add(file)
								
								var input = document.createElement("input")
									input.type = "file"
									input.name = "file"
									input.files = container.files
								
								form.appendChild(input)
							}
						}
						
						if(form.enctype){
							document.head.appendChild(form)
							document.head.appendChild(iframe)

							if(req.url.indexOf('amazonaws.com') > -1){
								OAuth3.network.shift()
							}

							form.submit()

							form.outerHTML = ""
						}
					}catch(err){
						return {error : err}
					}
				}

				req.abort = function(){
					var i = 0
					
					while (i < OAuth3.network.length) {
						if (OAuth3.network[i] === state) {
							OAuth3.network.splice(i, 1)
						} else {
							++i
						}
					}

					OAuth3.network[state] = undefined
					
					try{
						$('head iframe[name="'+state+'"]').remove()
					}catch(err){

					}
				}

				return req
			}
		}

	window.OAuth3 = OAuth3

	
	// window.addEventListener('blur', function(){
	// 	if(OAuth3.network){
	// 		if(OAuth3.network.length){
	// 			state = OAuth3.network[0]

	// 			if(OAuth3.network[state]){
	// 				if(OAuth3.network[state].request.mailto){
	// 					OAuth3.fetch()
	// 				}
	// 			}
	// 		}
	// 	}
	// })
	
	window.addEventListener("message", function(res){

		if(res.data){
			try{
				if(typeof res.data == "string"){
					if(OAuth3.network.length){
						res.body = JSON.parse(res.data)

						if(res.body.query){
							var state = res.body.query.state
							
							var delay = 0

							if(OAuth3.network[state]){
								OAuth3.network.shift()

								var response = OAuth3.network[state].response
								
								try{
									response(res)
								}catch(err){

								}

								try{
									var request = OAuth3.network[state].request
									
									if(request.delay){
										delay = request.delay

										if(!OAuth3.network[state].timeout){
											OAuth3.network[state].timeout = true
										}
									}else{
										OAuth3.network[state] = undefined
									}
								}catch(err){

								}
							}
							
							if(OAuth3.network.length){
								if(OAuth3.network[state]){
									if(delay && OAuth3.network[state].timeout){
										clearTimeout(OAuth3.network[state].timeout)
										OAuth3.network[state].timeout = setTimeout(OAuth3.fetch, delay)
									}else{
										OAuth3.fetch()
									}
								}else{
									OAuth3.fetch()
								}
							}else if(OAuth3.network[state]){
								if(OAuth3.network[state].timeout){
									clearTimeout(OAuth3.network[state].timeout)
								}
								OAuth3.network[state] = undefined
							}

							try{
								$('head iframe[name="'+state+'"]').remove()
							}catch(err){

							}
						}
					}
				}
			}catch(err){
				console.log("task timeout", err)
			}
		}
	}, false)
})()