// 이미지 · 검색 갈래(Rust: workers/proxy/src/image.rs · search.rs · rules.rs · handler.rs)의 버그 수정을 원본 TS 에 덧댄다.
// 같은 id 의 Rust 수정과 1:1 — 줄 번호는 원본(801b0ce0) 기준. fix-sql-params 는 Rust 만(시험 자료에 따옴표가 없어 동작이 같다).
export default [
  // 2568: `var item` 이 앞 회차의 item 을 남긴다 → 회차마다 비운다
  {
    id: "fix-image-item-reset", worker: "proxy",
    find: `var item

								if(models['deepinfra']){
									item = await Deepinfra(deepinfra, 'Qwen/Qwen3.5-0.8B', system, '', null, inlineData)`,
    replace: `var item = undefined

								if(models['deepinfra']){
									item = await Deepinfra(deepinfra, 'Qwen/Qwen3.5-0.8B', system, '', null, inlineData)`,
  },
  // 2578: Gemini 대체 호출이 config = null 을 넘겨 요청 뒤 config["response_mime_type"] 에서 던진다 → undefined(기본 JSON 설정)
  {
    id: "fix-image-gemini-config", worker: "proxy",
    find: "item = await Gemini(gemini_llm_api, gemini_llm_model, '', system, null, inlineData)",
    replace: "item = await Gemini(gemini_llm_api, gemini_llm_model, '', system, undefined, inlineData)",
  },
  // 2738: pages[cc].order 가 없으면 던진다 → tracking 칸처럼 먼저 만든다
  {
    id: "fix-image-order-slot", worker: "proxy",
    find: `team.data.base.pages[task.cc].order.draft--
											team.data.base.pages[task.cc].order.count++`,
    replace: `if(!team.data.base.pages[task.cc].order){
												team.data.base.pages[task.cc].order = {
													draft : 0,
													count : 0
												}
											}

											team.data.base.pages[task.cc].order.draft--
											team.data.base.pages[task.cc].order.count++`,
  },
  // 2868 · 2890 (CLIENT_EMBEDDING 이 참이라 죽은 갈래): embeddings 를 비우고, Deepinfra 의 벡터 하나를 수마다 map 하지 않는다
  {
    id: "fix-image-embeddings", worker: "proxy",
    find: `var embeddings

											if(models['cloudflare']){
												var { data: embeddings } = await env.AI.run('@cf/google/embeddinggemma-300m', {
													text: [item.text]
												})`,
    replace: `var embeddings = undefined

											if(models['cloudflare']){
												var { data: embeddings } = await env.AI.run('@cf/google/embeddinggemma-300m', {
													text: [item.text]
												})`,
  },
  {
    id: "fix-image-embeddings", worker: "proxy",
    find: `var embeddings = await Deepinfra(deepinfra, 'google/embeddinggemma-300m', '', item.text)

												var $VectorizeVector: VectorizeVector[] = embeddings.map((values, i) => {
													return {
														id: item.id,
														values: values,
														metadata: metadata
													}
												})`,
    replace: `var embeddings = await Deepinfra(deepinfra, 'google/embeddinggemma-300m', '', item.text)

												var $VectorizeVector: VectorizeVector[] = [
													{
														id: item.id,
														values: embeddings,
														metadata: metadata
													}
												]`,
  },
  // 2911~2939: items 넣기가 task 의 칸을 묶고(task.digest 가 없으면 D1_TYPE_ERROR) ON CONFLICT 가 ref 를 고치지 않는다
  {
    id: "fix-image-items-bind", worker: "proxy",
    find: `"bcc" = EXCLUDED."bcc",
													"digest" = EXCLUDED."digest",
													"data" = EXCLUDED."data",
													"created_at" = EXCLUDED."created_at",
													"updated_at" = EXCLUDED."updated_at"
											\`).bind(
												task.id,
												task.type,
												task.from,
												task.to,
												task.cc,
												task.bcc,
												task.ref,
												task.digest,`,
    replace: `"bcc" = EXCLUDED."bcc",
													"ref" = EXCLUDED."ref",
													"digest" = EXCLUDED."digest",
													"data" = EXCLUDED."data",
													"created_at" = EXCLUDED."created_at",
													"updated_at" = EXCLUDED."updated_at"
											\`).bind(
												item.id,
												item.type,
												item.from,
												item.to,
												item.cc,
												item.bcc,
												item.ref,
												item.digest,`,
  },
  // 5706 · 5770 · 5951 · 6225: 다시 선언해도 지워지지 않는 var — 작업(paragraphs · contexts) · 문맥(queryVector · generation)마다 비운다
  {
    id: "fix-search-reset", worker: "proxy",
    find: `var paragraphs

								var system = para2graph(language)`,
    replace: `var paragraphs = undefined

								var system = para2graph(language)`,
  },
  {
    id: "fix-search-reset", worker: "proxy",
    find: `var contexts

								var system = graph2contexts(current)`,
    replace: `var contexts = undefined

								var system = graph2contexts(current)`,
  },
  {
    id: "fix-search-reset", worker: "proxy",
    find: `var queryVector

											console.log('context.text',context.text);`,
    replace: `var queryVector = undefined

											console.log('context.text',context.text);`,
  },
  {
    id: "fix-search-reset", worker: "proxy",
    find: `var generation

											if(models['deepinfra']){
												generation = await Deepinfra(deepinfra, 'openai/gpt-oss-20b', system, content)`,
    replace: `var generation = undefined

											if(models['deepinfra']){
												generation = await Deepinfra(deepinfra, 'openai/gpt-oss-20b', system, content)`,
  },
  // 5759 · 5765: 배열이 아닌 paragraphs(글 등)는 rowsTrim 이 던지고, length 가 없으면 그대로 지나간다 → 빈 경우처럼
  {
    id: "fix-search-paragraphs-array", worker: "proxy",
    find: `if(paragraphs.length){
									paragraphs = rowsTrim(paragraphs)
								}`,
    replace: `if(Array.isArray(paragraphs) && paragraphs.length){
									paragraphs = rowsTrim(paragraphs)
								}`,
  },
  {
    id: "fix-search-paragraphs-array", worker: "proxy",
    find: `if(paragraphs.length == 0){
									continue
								}`,
    replace: `if(!Array.isArray(paragraphs) || paragraphs.length == 0){
									continue
								}`,
  },
  // 5782~5789 · 5881: LLM 이 글 · 수를 돌려주면 속성 쓰기가 (엄격 모드라) 던진다 → 객체일 때만 쓴다
  {
    id: "fix-search-primitive-results", worker: "proxy",
    find: `if(!results.text){
												results.text = task.body
											}

											if(!results.type && paragraphs.length == 1){
												results.type = paragraphs[0].type
											}`,
    replace: `if(typeof results == 'object'){
												if(!results.text){
													results.text = task.body
												}

												if(!results.type && paragraphs.length == 1){
													results.type = paragraphs[0].type
												}
											}`,
  },
  {
    id: "fix-search-primitive-results", worker: "proxy",
    find: "context.id = hashId()",
    replace: "if(context && typeof context == 'object'){ context.id = hashId() }",
  },
  // 6020: context.condition 이 없으면 Object.keys 가 던진다 → 객체가 아니면 {}
  {
    id: "fix-search-condition-guard", worker: "proxy",
    find: "if(Object.keys(context.condition).length){",
    replace: "if(Object.keys(context.condition && typeof context.condition == 'object' ? context.condition : {}).length){",
  },
  // 6088 · 6092: matches 에 length 가 없으면 던진다 → 0
  {
    id: "fix-search-matches-length", worker: "proxy",
    find: "console.log('matches.length',matches.length.toString());",
    replace: "console.log('matches.length',((matches && matches.length) || 0).toString());",
  },
  {
    id: "fix-search-matches-length", worker: "proxy",
    find: `if(matches.length){
												for(var m = 0; m < matches.length; m++){`,
    replace: `if(matches && matches.length){
												for(var m = 0; m < matches.length; m++){`,
  },
  // 6130~6135: 함수 범위의 i 로 지운다 → r
  {
    id: "fix-search-delete-index", worker: "proxy",
    find: `delete results[i].from
															delete results[i].to
															delete results[i].cc
															delete results[i].bcc
															delete results[i].ref
															delete results[i].data`,
    replace: `delete results[r].from
															delete results[r].to
															delete results[r].cc
															delete results[r].bcc
															delete results[r].ref
															delete results[r].data`,
  },
  // 6161: created_at 변수(수)를 찾아 사실상 붙지 않는다 → 열 이름을 찾는다
  {
    id: "fix-search-created-at", worker: "proxy",
    find: "if(condition.indexOf(created_at) > -1){",
    replace: `if(condition.indexOf('"created_at"') > -1){`,
  },
];
