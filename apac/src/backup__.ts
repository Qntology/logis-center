import { gzip, ungzip } from 'pako'

async function Sleep(ms) {
	return new Promise(resolve => setTimeout(resolve, ms))
}

const CenterRegion = "logis_central"

async function Cron(event, env, ctx, models, limits, delay){
	/*
		매월 1일에 결제한 사용자를 기준으로 
		사용 가능한 balance 지급하는 프로세스 추가해야함
	*/
	var now = Date.now()
	
	var created_at = now

	try{
		var { results } = await env[env.region].prepare(`SELECT * FROM tasks WHERE "created_at" < ${created_at} AND "updated_at" = 0 ORDER BY created_at ASC LIMIT 1000`).all()

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

				if(limits[task.id]){
					len--
					continue
				}

				if(task.method){
					delete task.method
				}

				tasks.push(task)
			}


			if(tasks.length){
				for(var t = 0; t < tasks.length; t++){
					var task = tasks[t]

					if(limits[task.id]){
						continue
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

						limits[task.id] = true

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

					await Sleep(300 * delay)
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


		var startTime = Date.now();
	    
	    // 2. 최대 실행 시간 설정 (55초)
	    // 1분(60초) 스케줄러가 다시 실행되기 전에 종료하여 중복을 피합니다.
	    var MAX_RUN_TIME_MS = 55 * 1000; // 55,000 밀리초

		var delay = 0.3


		while(true){
			var elapsedTime = Date.now() - startTime;
			var timeLeft = MAX_RUN_TIME_MS - elapsedTime;

			if (timeLeft <= 500) { // 남은 시간이 0.5초(500ms) 이하이면 종료
				break; 
			}

			var results = await Cron(event, env, ctx, models, limits, delay)

			limits = results.limits



			models = results.models

			if(results.length){
				MAX_RUN_TIME_MS = MAX_RUN_TIME_MS - (results.length * 2000)
				delay = 0.3
			}else{
				delay += 0.3
			}


			await Sleep(300 * delay)
		}
	}
} satisfies ExportedHandler<Env>