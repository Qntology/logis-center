import { gzip, ungzip } from 'pako'

async function Sleep(ms) {
	return new Promise(resolve => setTimeout(resolve, ms))
}

const CenterRegion = "center_logis"

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

		var tasks = []

		var clear_condition = ""

		if (len) {
			console.log('tasks len',len)
			
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
								first :'gemini-2.0-flash-lite',
								second:'gemini-2.5-flash-lite'
							}
						}else{
							return {
								first :'gemini-2.5-flash-lite',
								second:'gemini-2.0-flash-lite'
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


					const res = await fetch(`https://proxy.logis.center`, {
						method: "POST",
						headers: {
							'Content-Type': 'application/octet-stream',
							'Content-Encoding': 'gzip'
						},
						body: arr.buffer
					});

					try{
						var results = await res.json();

						models = results.models
						limits = results.limits

						pageCount = results.counts

					}catch(err){
						console.log('err',err);
					}
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

			await Sleep(1000 * delay)
		}
	}
} satisfies ExportedHandler<Env>