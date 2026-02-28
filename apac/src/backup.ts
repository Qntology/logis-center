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

const CenterRegion = "logis_central"

async function Cron(event, env, ctx, models, limits, timeout){
	/*
		매월 1일에 결제한 사용자를 기준으로 
		사용 가능한 balance 지급하는 프로세스 추가해야함
	*/
	var now = Date.now()
	
	var created_at = now

	try{
		var { results } = await env[env.region].prepare(`SELECT * FROM tasks WHERE "created_at" < ${created_at} AND "updated_at" = 0 ORDER BY created_at ASC LIMIT 100`).all()

		var len = results.length

		var tasks = []

		var clear_condition = ""

		if (len) {
			console.log('limits',JSON.stringify(limits));
			console.log('tasks len',len)
			
			for(var i = 0; i < results.length; i++){
				var cron = results[i]

				var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(cron.task))

				var task = JSON.parse(decompressedJsonString)

				if(task.method){
					delete task.method
				}

				tasks.push(task)
			}


			if(tasks.length){
				for(var t = 0; t < tasks.length; t++){
					var task = tasks[t]

					await Sleep(500 * timeout.delay)

					var elapsedTime = Date.now() - timeout.start;
					var timeLeft = timeout.max - elapsedTime;

					if (timeLeft <= 500) { // 남은 시간이 0.5초(500ms) 이하이면 종료
						break; 
					}

					var { results } = await env[env.region].prepare(`SELECT * FROM tasks WHERE "created_at" < ${task.created_at} AND "updated_at" > 0 ORDER BY created_at ASC LIMIT 1`).all()

					console.log('results.length',results.length);

					var _cron = tasks[t-1]

					if(results.length){
						var _cron = results[0]

						var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(_cron.data))

						var _task = JSON.parse(decompressedJsonString)

						var before = _task.team

						console.log('before team.data',JSON.stringify(before.data));


						var logisRegion = LogisRegion[_task.flag]



						var { results } = await env[logisRegion].prepare(`SELECT * FROM users WHERE "type" = 'team' AND "id" = '${before.id}' AND "created_at" < ${now} LIMIT 1`).all()

						var after = results[0]

						if(after.data){
							var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(after.data))

							after.data = JSON.parse(decompressedJsonString)
						}else{
							after.data = {}
						}

						console.log('isDiff(before.data, after.data)',isDiff(before.data, after.data));

						if(isDiff(before.data, after.data)){
							break
						}else{
							limits[_cron.id] = true

							await env[env.region].prepare(`
								DELETE FROM tasks WHERE "id" = '${_cron.id}'
							`).run()
						}
					}else if(_cron){
						if(!limits[_cron.id]){
							console.log('stop 진입');

							break
						}else{
							console.log('통과');
						}
					}else if(t == 0 && tasks.length > 1){
						console.log('미정');

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
						clear_condition += ` AND "id" != "${task.id}"`

						await env[region].prepare(`
							DELETE FROM tasks WHERE id = "${task.id}"
						`).run()

						continue
					}

					

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


					const res = await fetch(`https://proxy.logis.center`, {
						method: "POST",
						headers: {
							'Content-Type': 'application/octet-stream',
							'Content-Encoding': 'gzip'
						},
						body: arr.buffer
					});

					try{
						var _results = await res.json();

						models = _results.models
						limits = _results.limits

					}catch(err){
						console.log('err',err);
					}
				}
			}


		}else{
			timeout.delay++
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
			delay : 1,
			start : Date.now(),
			max : 55 * 1000
		}

		while(true){
			var elapsedTime = Date.now() - timeout.start;
			var timeLeft = timeout.max - elapsedTime;

			if (timeLeft <= 500) { // 남은 시간이 0.5초(500ms) 이하이면 종료
				break; 
			}

			var results = await Cron(event, env, ctx, models, limits, timeout)

			limits = results.limits

			models = results.models

			if(results.length){
				timeout.delay = 1
			}else{
				timeout.delay++
			}

			await Sleep(500 * timeout.delay)
		}
	}
} satisfies ExportedHandler<Env>