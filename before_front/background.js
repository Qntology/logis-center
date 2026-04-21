importScripts('dexie.min.js');

importScripts('pako.min.js');




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
	users : `
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


chrome.runtime.onMessage.addListener(function(req, sender, res) {
	(async function() {
		if (req.url) {
			try {
				var option = {
					method: req.method,
					headers: req.headers
				}

				if(req.headers['Content-Encoding'] == 'gzip'){
					try{
						if(req.body){
							if(Object.keys(req.body).length){
								var arr = pako.gzip(new TextEncoder('utf-8').encode(req.body), { to: 'arraybuffer' })

								option.body = arr.buffer
							}
						}
					}catch(err){

					}
						
				}

				var response = await fetch(req.url, option)

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
									try{
										var arr = new Uint8Array(item.data)

										var decompressedJsonString = new TextDecoder('utf-8').decode(pako.ungzip(arr.buffer))

										var data = JSON.parse(decompressedJsonString)

									}catch(err){

									}

								}

								json.results[i].data = data
							}
						}
					}
				}

				res({ json : json })
			} catch (error) {
				console.log('error',error);
				res({ json : {
					error : error.message
				} })
			}
		}else{
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
					results = await db[req.delete].where(req.key).equals(req.value).delete();
					
				}else if(req.clear){
					results = await db[req.clear].clear();

				}

				res({ results : results });

			}catch(error) {
				res({ error: error });
			}
		}
			
	})();

	return true;
});