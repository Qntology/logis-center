// 아이템 반복 3 · 4 구역(원본 4556~5516, Rust item3.rs · item4.rs)의 버그 수정 — Rust 의 같은 수정과 1:1.
// fix-sql-params 는 Rust 만(시험 데이터에 작은따옴표가 없어 원본의 글 이어 붙이기와 결과가 같다).
export default [
  {
    // 4594: var merge 는 값을 지우지 않아 update · upsert 가 없으면 앞 행(또는 4471 Relay)의 merge 가 남았다
    id: "fix-relate-merge-reset", worker: "proxy",
    find: `var merge

		if(relate.merge.update){`,
    replace: `var merge = undefined

		if(relate.merge.update){`,
  },
  {
    // 4633~4640: 풀기에 실패하면 앞 행의 data 가 남았다 → try 앞에서 지운다
    id: "fix-relate-data-reset", worker: "proxy",
    find: `try{
			var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(from.data))`,
    replace: `var data = undefined

		try{
			var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(from.data))`,
  },
  {
    // 4680: data 가 없으면 data.origin 이 던졌다 → 빈 값과 같은 기본값
    id: "fix-relate-data-reset", worker: "proxy",
    find: `origin : data.origin ? data.origin : "",`,
    replace: `origin : data && data.origin ? data.origin : "",`,
  },
  {
    // 4689
    id: "fix-relate-data-reset", worker: "proxy",
    find: `title : data.title ? data.title : "",`,
    replace: `title : data && data.title ? data.title : "",`,
  },
  {
    // 4708: var semantic 이 앞 행의 값을 지니고 있었다
    id: "fix-relate-semantic-reset", worker: "proxy",
    find: `var content = JSON.stringify(metadata)

		var semantic

		var system = semantic_prompt_system(language).trim()`,
    replace: `var content = JSON.stringify(metadata)

		var semantic = undefined

		var system = semantic_prompt_system(language).trim()`,
  },
  {
    // 4746 · 4748 (CLIENT_EMBEDDING 이 참이라 지금은 죽은 갈래)
    id: "fix-relate-semantic-reset", worker: "proxy",
    find: `var embeddings

		var $VectorizeVector

		if(models['cloudflare']){`,
    replace: `var embeddings = undefined

		var $VectorizeVector = undefined

		if(models['cloudflare']){`,
  },
  {
    // 4782: 조건이 뒤집혀 임베딩이 있을 때 건너뛰었다 (죽은 갈래)
    id: "fix-embeddings-continue", worker: "proxy",
    find: `if(embeddings){
			fallback = 'embeddings overflow'`,
    replace: `if(!embeddings){
			fallback = 'embeddings overflow'`,
  },
  {
    // 4768: .tirm() 오타 (죽은 갈래)
    id: "fix-tirm", worker: "proxy",
    find: `semantic.tirm())

		var $VectorizeVector: VectorizeVector[] = embeddings.map((values, i) => {
			return {
				id: edge.id,`,
    replace: `semantic.trim())

		var $VectorizeVector: VectorizeVector[] = embeddings.map((values, i) => {
			return {
				id: edge.id,`,
  },
  {
    // 5210: .tirm() 오타 (죽은 갈래)
    id: "fix-tirm", worker: "proxy",
    find: `item.semantic.tirm()`,
    replace: `item.semantic.trim()`,
  },
  {
    // 4797: try 밖에서 ungzip — to 가 item 이면 data 가 이미 객체라 던져 스캔이 멈췄다 → 바이트일 때만 푼다
    id: "fix-edge-data-ungzip", worker: "proxy",
    find: `var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(edge.data))

		edge.data = JSON.parse(decompressedJsonString)`,
    replace: `if(edge.data instanceof ArrayBuffer || ArrayBuffer.isView(edge.data) || Array.isArray(edge.data)){
			var decompressedJsonString = new TextDecoder('utf-8').decode(ungzip(edge.data))

			edge.data = JSON.parse(decompressedJsonString)
		}`,
  },
  {
    // 5046~5068 (목록 밖 추가 수정): to 가 이어진 행이면(이벤트 · 주문 등) pages[edge.cc][edge.type] 칸이 없어 .draft 에서 던졌다
    // → 셈 앞에 다른 곳(3280~3297)처럼 칸을 만든다
    id: "fix-edge-slots", worker: "proxy",
    find: `var { results } = await env[\`commerce_logis_\${zoneRegion}_items\`].prepare(\`SELECT * FROM items WHERE "id" = '\${edge.id}'`,
    replace: `if(!team.data.base.pages[edge.cc]){
			team.data.base.pages[edge.cc] = {}
		}

		if(!team.data.base.pages[edge.cc][edge.type]){
			team.data.base.pages[edge.cc][edge.type] = {
				draft : 0,
				count : 0
			}
		}

		if(!team.data.base[edge.type]){
			team.data.base[edge.type] = {
				draft : 0,
				count : 0
			}
		}

		var { results } = await env[\`commerce_logis_\${zoneRegion}_items\`].prepare(\`SELECT * FROM items WHERE "id" = '\${edge.id}'`,
  },
  {
    // 5062: row.id 가 아니라 방금 찾은 items 행(SELECT … WHERE id = edge.id)을 고친다
    id: "fix-undefined-binds", worker: "proxy",
    find: `UPDATE items SET updated_at = ? WHERE id = ?
		\`).bind(
			now, row.id
		)`,
    replace: `UPDATE items SET updated_at = ? WHERE id = ?
		\`).bind(
			now, edge.id
		)`,
  },
  {
    // 5102: item.id 가 아니라 이 관계의 초안 행(edge.id — 아래 else 가 그 id 로 등록한다)을 고친다
    id: "fix-undefined-binds", worker: "proxy",
    find: `UPDATE items SET updated_at = ? WHERE id = ?
		\`).bind(
			now, item.id
		)`,
    replace: `UPDATE items SET updated_at = ? WHERE id = ?
		\`).bind(
			now, edge.id
		)`,
  },
  {
    // 5079~5087: 이어진 행이 없으면 앞 회차의 edgeType · edge 를 썼다(처음이면 env["commerce_logis_<zone>_undefined"] 가 없어 던져
    // 스캔이 첫 아이템에서 멈춤) → 이 관계의 질의로 관계 유형의 초안을 찾거나 등록한다
    id: "fix-edge-draft", worker: "proxy",
    find: `// var draftId = hashId(item.id)`,
    replace: `var q = query[1] ? query[1] : query[0]

		edgeType = q.table

		edge = {
			id : hashId(type + item.id),
			type : type,
			cc : item.cc,
			index : q.value
		}

		if(!team.data.base.pages[edge.cc]){
			team.data.base.pages[edge.cc] = {}
		}

		if(!team.data.base.pages[edge.cc][edge.type]){
			team.data.base.pages[edge.cc][edge.type] = {
				draft : 0,
				count : 0
			}
		}

		if(!team.data.base[edge.type]){
			team.data.base[edge.type] = {
				draft : 0,
				count : 0
			}
		}`,
  },
  {
    id: "fix-item4-embeddings-reset", worker: "proxy",
    find: "console.log('[EMBED-PENDING] item delegated to Client App:', item.id) }else{ var embeddings if(models['cloudflare']){",
    replace: "console.log('[EMBED-PENDING] item delegated to Client App:', item.id)\n}else{\nvar embeddings = undefined\n\nif(models['cloudflare']){",
  },
];
