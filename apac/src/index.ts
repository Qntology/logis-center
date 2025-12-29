import { gzip, ungzip } from 'pako'

async function Sleep(ms) {
	return new Promise(resolve => setTimeout(resolve, ms))
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


async function Cron(event, env, ctx, models, limits, timeout){
	/*
		매월 1일에 결제한 사용자를 기준으로 
		사용 가능한 balance 지급하는 프로세스 추가해야함
	*/
	var now = Date.now()
	
	var created_at = now

	try{
		var { results } = await env[env.region].prepare(`SELECT * FROM tasks WHERE "created_at" < ${created_at} ORDER BY created_at ASC LIMIT 300`).all()

		var len = results.length

		var tasks = []

		if (len) {
			console.log('limits',JSON.stringify(limits));
			console.log('tasks len',len)
			
			for(var i = 0; i < len; i++){
				var cron = results[i]

				var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(cron.data))

				var task = JSON.parse(decompressedJsonString)

				if(cron.id == 'lock'){
					tasks = []

					break;
				}

				if(task.method){
					delete task.method
				}

				tasks.push(task)
			}


			if(tasks.length){
				for(var t = 0; t < tasks.length; t++){
					var task = tasks[t]

					await Sleep(timeout.value * timeout.delay)

					var elapsedTime = Date.now() - timeout.start;
					var timeLeft = timeout.max - elapsedTime;

					if (timeLeft <= 5000) { // 남은 시간이 0.5초(500ms) 이하이면 종료
						break; 
					}

					var { results } = await env[env.region].prepare(`SELECT * FROM tasks WHERE "created_at" <= ${task.created_at} AND "updated_at" > 0 ORDER BY created_at ASC LIMIT 1`).all()

					if(results){
						console.log('results.length',results.length);

						var _cron = tasks[t-1]

						if(results.length){
							var _cron = results[0]

							var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(_cron.data))

							var _task = JSON.parse(decompressedJsonString)

							var before = _task.team

							var logisRegion = LogisRegion[_task.flag]

							var { results } = await env[logisRegion].prepare(`SELECT * FROM users WHERE "type" = 'team' AND "id" = '${before.id}' AND "created_at" < ${now} LIMIT 1`).all()

							var after = results[0]

							var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(after.data))

							after.data = JSON.parse(decompressedJsonString)

							var diff = isDiff(before.data, after.data)

							console.log('diff',diff);

							if(diff){
								timeout.delay += 0.5

								break
							}else{
								console.log('1.remove 2.통과 진입');
								timeout.delay = 0.5
								

								await env[env.region].prepare(`
									DELETE FROM tasks WHERE "id" = '${_cron.id}'
								`).run()

								if(_cron.id == task.id){
									console.log('self stop 진입');
									break
								}else{
									limits[_cron.id] = true

									limits.team = before
								}
							}
						}else if(_cron){
							if(!limits[_cron.id]){
								timeout.delay += 0.5

								console.log('stop 진입');

								break
							}else{
								timeout.delay = 1
								console.log('통과');
							}
						}else if(t == tasks.length - 1 && limits[task.id.toUpperCase()]){
							console.log('미정');
							timeout.delay += 0.5

							break
						}


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
									first :'gemini-flash-lite-latest',
									second:'gemini-flash-lite-latest'
								}
							}else{
								return {
									first :'gemini-flash-lite-latest',
									second:'gemini-flash-lite-latest'
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

						}else if(!models['deepinfra']){
							await env[env.region].prepare(`
								DELETE FROM tasks WHERE id = "${task.id}"
							`).run()

							continue
						}


						// lock
						var { results, success, error } =  await env[env.region].prepare(`
							INSERT INTO tasks (
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
							'lock',
							'',
							'',
							'',
							'',
							'',
							'',
							null,
							1,
							0
						).run()

						

						var arr = gzip(new TextEncoder('utf-8').encode(JSON.stringify({
							now : now,
							id : task.id,
							ref : task.ref,
							region : env.region,
							models : models,
							limits : limits,
							gemini_llm_api : gemini_llm_api,
							gemini_llm_model : gemini_llm_model,
							deepinfra : env.deepinfra
						})), { to: 'arraybuffer' })

						try{
							const res = await fetch(`https://proxy.commerce.logis.center`, {
								method: "POST",
								headers: {
									'Content-Type': 'application/octet-stream',
									'Content-Encoding': 'gzip'
								},
								body: arr.buffer
							});

							var _results = await res.json();

							models = _results.models
							limits = _results.limits

						}catch(err){
							await env[env.region].prepare(`
								DELETE FROM tasks WHERE "id" = 'lock'
							`).run()
							console.log('err',err);
						}
					}

				}
			}


		}else{
			timeout.delay += 0.5
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

		var timeout = {
			value : 1000,
			delay : 0.5,
			start : Date.now(),
			max : 50 * 1000
		}

		while(true){
			var elapsedTime = Date.now() - timeout.start;
			var timeLeft = timeout.max - elapsedTime;

			if (timeLeft <= 5000) { // 남은 시간이 0.5초(500ms) 이하이면 종료
				break; 
			}

			var results = await Cron(event, env, ctx, models, limits, timeout)

			limits = results.limits

			models = results.models

			if(results.length){
				timeout.delay = 0.5
			}else{
				timeout.delay += 0.5
			}

			await Sleep(timeout.value * timeout.delay)
		}
	}
} satisfies ExportedHandler<Env>