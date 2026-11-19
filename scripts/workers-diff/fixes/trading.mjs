// trading 원본(trading-logis-center 65684e6)에 덧대는 수정 — Rust 판(workers/trading)의 `// 고침 (fix-trading-…)` 과 1:1.
// id 는 docs/workers-rs.md 의 "고친 버그" 표와 같다. 차례대로 적용되므로 뒤 항목의 find 는 앞 항목이 바꾼 글을 본다.
const L = (...lines) => lines.join("\n");
const TOP = "const TRADING_MODE = 'shipping'"; // 최상위 상수 · 함수를 덧붙이는 자리 (늘 이 줄을 앞에 남긴다)

export default [
  // ── request.cf 가 없으면 try 밖에서 TypeError (처리기가 통째로 던짐) ──
  { id: "fix-trading-cf-optional", worker: "trading", find: "var { country } = request.cf", replace: "var { country } = request.cf ? request.cf : {}" },

  // ── 평범한 객체 조회에 Object.prototype 의 이름이 섞임 ──
  {
    id: "fix-trading-proto-keys", worker: "trading",
    find: "var language = languageCode[FLAG] ? languageCode[FLAG] : 'English'",
    replace: "var language = Object.prototype.hasOwnProperty.call(languageCode, FLAG) ? languageCode[FLAG] : 'English'",
  },
  {
    id: "fix-trading-proto-keys", worker: "trading",
    find: "cookies.hello = Hello[language] ? Hello[language] : Hello['English']",
    replace: "cookies.hello = Object.prototype.hasOwnProperty.call(Hello, language) ? Hello[language] : Hello['English']",
  },
  { id: "fix-trading-proto-keys", worker: "trading", find: "var brief: any = {}", replace: "var brief: any = Object.create(null)" },
  { id: "fix-trading-proto-keys", worker: "trading", find: "var seenId = {}", replace: "var seenId = Object.create(null)" },
  { id: "fix-trading-proto-keys", worker: "trading", find: "var existing = {}", replace: "var existing = Object.create(null)" },
  { id: "fix-trading-proto-keys", worker: "trading", find: "var hubBrief = {}", replace: "var hubBrief = Object.create(null)" },

  // ── Cookie 머리글의 address · team 이 인증된 신원으로 쓰임 ──
  {
    id: "fix-trading-cookie-identity", worker: "trading",
    find: L("if(parts.length === 2){", "cookies[parts[0].trim()] = parts[1].trim()", "}"),
    replace: L("if(parts.length === 2 && (parts[0].trim() == 'hash' || parts[0].trim() == 'token')){", "cookies[parts[0].trim()] = parts[1].trim()", "}"),
  },

  // ── preflight 가 세션 뒤에 있어 OPTIONS 마다 bootstrap S3 PUT ──
  {
    id: "fix-trading-options-first", worker: "trading",
    find: L(
      "var requestOrigin = request.headers.get('Origin')",
      "headers.set('Access-Control-Allow-Origin', requestOrigin ? requestOrigin : '*')",
      "headers.set('Access-Control-Allow-Credentials', 'true')",
      "headers.set('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')",
      "headers.set('Access-Control-Allow-Headers', 'Content-Type, Content-Encoding, Authorization, X-Requested-With')",
      "if(request.method == 'OPTIONS'){",
      "return new Response('', { headers: headers })",
      "}",
    ),
    replace: "",
  },
  {
    id: "fix-trading-options-first", worker: "trading",
    find: "// ── 인증 : commerce / console 과 완전히 같은 S3 hash/token 계약 ──",
    replace: L(
      "var requestOrigin = request.headers.get('Origin')",
      "headers.set('Access-Control-Allow-Origin', requestOrigin ? requestOrigin : '*')",
      "headers.set('Access-Control-Allow-Credentials', 'true')",
      "headers.set('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')",
      "headers.set('Access-Control-Allow-Headers', 'Content-Type, Content-Encoding, Authorization, X-Requested-With')",
      "if(request.method == 'OPTIONS'){",
      "return new Response('', { headers: headers })",
      "}",
      "",
      "// ── 인증 : commerce / console 과 완전히 같은 S3 hash/token 계약 ──",
    ),
  },

  // ── 아무 Origin 이나 Allow-Credentials 와 함께 반사 ──
  {
    id: "fix-trading-cors", worker: "trading",
    find: TOP,
    replace: L(TOP, "const ALLOWED_ORIGINS = ['https://logis.center', 'https://commerce.logis.center', 'https://trading.logis.center']"),
  },
  {
    id: "fix-trading-cors", worker: "trading",
    find: L("headers.set('Access-Control-Allow-Origin', requestOrigin ? requestOrigin : '*')", "headers.set('Access-Control-Allow-Credentials', 'true')"),
    replace: L(
      "if(!requestOrigin){",
      "headers.set('Access-Control-Allow-Origin', '*')",
      "headers.set('Access-Control-Allow-Credentials', 'true')",
      "}else if(ALLOWED_ORIGINS.indexOf(requestOrigin) > -1){",
      "headers.set('Access-Control-Allow-Origin', requestOrigin)",
      "headers.set('Access-Control-Allow-Credentials', 'true')",
      "}",
    ),
  },

  // ── token 이 둘 다 없으면 null == undefined 로 hash 만으로 통과 ──
  {
    id: "fix-trading-token-required", worker: "trading",
    find: "if(email && token == cookies.token){",
    replace: "if(email && token && cookies.token && token == cookies.token){",
  },

  // ── 확인하지 않은 hash · token 을 Set-Cookie 로 되돌림 (쿠키 속성 끼워 넣기) ──
  {
    id: "fix-trading-setcookie-verified", worker: "trading",
    find: L(
      "console.log(`[TRADING] ⛔ 팀 확인 실패로 이 요청은 데이터 접근이 차단됩니다. (address='${address}')`)",
      "}",
      "}",
      "headers.set('Set-Cookie', `hash=${cookies.hash}; Secure; HttpOnly; SameSite=None`)",
      "headers.append('Set-Cookie', `token=${cookies.token}; Secure; HttpOnly; SameSite=None`)",
      "if(cookies.team){",
      "headers.append('Set-Cookie', `team=${cookies.team}; Secure; HttpOnly; SameSite=None`)",
      "}",
    ),
    replace: L(
      "console.log(`[TRADING] ⛔ 팀 확인 실패로 이 요청은 데이터 접근이 차단됩니다. (address='${address}')`)",
      "}",
      "headers.set('Set-Cookie', `hash=${cookies.hash}; Secure; HttpOnly; SameSite=None`)",
      "headers.append('Set-Cookie', `token=${cookies.token}; Secure; HttpOnly; SameSite=None`)",
      "if(cookies.team){",
      "headers.append('Set-Cookie', `team=${cookies.team}; Secure; HttpOnly; SameSite=None`)",
      "}",
      "}",
    ),
  },

  // ── 응답 본문(session · HTML)에 token ──
  {
    id: "fix-trading-token-echo", worker: "trading",
    find: L("console.log('[TRADING] bootstrap err', err)", "}", "}"),
    replace: L("console.log('[TRADING] bootstrap err', err)", "}", "}", "delete cookies.token"),
  },

  // ── <pre> 에 JSON 을 그대로 (반사 XSS) ──
  {
    id: "fix-trading-html-escape", worker: "trading",
    find: TOP,
    replace: L(
      TOP,
      "function escapeHtml(s){",
      "return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\"/g, '&quot;').replace(/'/g, '&#39;')",
      "}",
    ),
  },
  {
    id: "fix-trading-html-escape", worker: "trading",
    find: "+ `<pre>${JSON.stringify(cookies, null, 2)}</pre>`",
    replace: "+ `<pre>${escapeHtml(JSON.stringify(cookies, null, 2))}</pre>`",
  },

  // ── 쿼리가 이미 풀린 값을 다시 decodeURIComponent ──
  {
    id: "fix-trading-double-decode", worker: "trading",
    find: "normalizeDocType(decodeURIComponent(req.query.doc_type))",
    replace: "normalizeDocType(req.query.doc_type)",
  },
  { id: "fix-trading-double-decode", worker: "trading", find: "pushDelId(decodeURIComponent(req.query.id))", replace: "pushDelId(req.query.id)" },
  { id: "fix-trading-double-decode", worker: "trading", find: "JSON.parse(decodeURIComponent(req.query.ids))", replace: "JSON.parse(req.query.ids)" },

  // ── NaN · Infinity 매개변수 ──
  {
    id: "fix-trading-nan-params", worker: "trading",
    find: L("if(!isNaN(sinceRaw)){", "since = parseInt(sinceRaw)", "}"),
    replace: L("if(!isNaN(sinceRaw)){", "since = parseInt(sinceRaw)", "}", "if(isNaN(since)){ since = 0 }"),
  },
  {
    id: "fix-trading-nan-params", worker: "trading",
    find: L("limit = Math.max(1, Math.min(1000, parseInt(req.query.limit)))", "}"),
    replace: L("limit = Math.max(1, Math.min(1000, parseInt(req.query.limit)))", "}", "if(isNaN(limit)){ limit = 1000 }"),
  },
  { id: "fix-trading-nan-params", worker: "trading", find: "if(!createdAt || isNaN(createdAt)){ createdAt = now }", replace: "if(!createdAt || !isFinite(createdAt)){ createdAt = now }" },
  { id: "fix-trading-nan-params", worker: "trading", find: "if(isNaN(updatedAt)){ updatedAt = 0 }", replace: "if(!isFinite(updatedAt)){ updatedAt = 0 }" },

  // ── id 를 trim · 소문자로 바꿔 저장 (앱은 그대로 씀) ──
  {
    id: "fix-trading-id-case", worker: "trading",
    find: L("var rid = list[p] && list[p].id ? String(list[p].id).trim().toLowerCase() : ''", "if(!rid){ continue }"),
    replace: L("var rid = list[p] && list[p].id ? String(list[p].id) : ''", "if(!rid.trim()){ continue }"),
  },
  {
    id: "fix-trading-id-case", worker: "trading",
    find: "var id = String(it.id).trim().toLowerCase()",
    replace: L("var id = String(it.id)", "if(!id.trim()){ rejected++; continue }"),
  },
  {
    id: "fix-trading-id-case", worker: "trading",
    find: L("var s = String(v ? v : '').trim().toLowerCase()", "if(!s){ return }"),
    replace: L("var s = String(v ? v : '')", "if(!s.trim()){ return }"),
  },

  // ── digest 사전 조회가 1 + id 수만큼 바인딩 (100 넘으면 D1 거절 → SKIP GUARD 조용히 꺼짐) ──
  //    (항목마다 따로 빼 볼 수 있게(OMIT_FIX) 다른 수정과 같은 글을 찾지 않는다)
  {
    id: "fix-trading-preload-chunk", worker: "trading",
    find: TOP,
    replace: L(TOP, "const PRELOAD_CHUNK = 99"),
  },
  {
    id: "fix-trading-preload-chunk", worker: "trading",
    find: L("if(idList.length){", "try{", "var ph = idList.map("),
    replace: L("for(var c0 = 0; c0 < idList.length; c0 += PRELOAD_CHUNK){", "var idChunk = idList.slice(c0, c0 + PRELOAD_CHUNK)", "try{", "var ph = idChunk.map("),
  },
  { id: "fix-trading-preload-chunk", worker: "trading", find: "...idList).all()", replace: "...idChunk).all()" },

  // ── ON CONFLICT (id) DO UPDATE 가 다른 팀의 행을 덮어씀 ──
  { id: "fix-trading-upsert-scope", worker: "trading", find: "return '?' + (n + 2)", replace: "return '?' + (n + 1)" },
  {
    id: "fix-trading-upsert-scope", worker: "trading",
    find: "`SELECT \"id\", \"digest\", \"ref\", \"updated_at\" FROM items WHERE \"to\" = ?1 AND \"id\" IN (${ph})`",
    replace: "`SELECT \"id\", \"to\", \"digest\", \"ref\", \"updated_at\" FROM items WHERE \"id\" IN (${ph})`",
  },
  { id: "fix-trading-upsert-scope", worker: "trading", find: ").bind(cookies.team, ...id", replace: ").bind(...id" },
  {
    id: "fix-trading-upsert-scope", worker: "trading",
    find: "var prev = existing[id]",
    replace: L(
      "var prev = existing[id]",
      "if(prev && prev.to != cookies.team){",
      "rejected++",
      "console.log(`[TRADING] ⛔ id='${id}' 은 다른 팀의 문서라 거부합니다.`)",
      "continue",
      "}",
    ),
  },
  {
    id: "fix-trading-upsert-scope", worker: "trading",
    find: L("\"updated_at\" = EXCLUDED.\"updated_at\"", "`).bind("),
    replace: L("\"updated_at\" = EXCLUDED.\"updated_at\"", "WHERE items.\"to\" = EXCLUDED.\"to\"", "`).bind("),
  },

  // ── 변경분을 클라이언트 시각으로 골라 커서보다 옛 시각으로 나중에 들어온 행을 잃음 · LIMIT 에 잘려도 커서를 now 로 ──
  //    (migrations/0002_add_synced_at.sql 이 synced_at 열을 더한다)
  {
    id: "fix-trading-delta-cursor", worker: "trading",
    find: L(
      "var where = `\"to\" = ?1 AND \"mode\" = ?2 AND \"created_at\" < ?3 AND max(\"updated_at\",\"created_at\") > ?4`",
      "var binds: any[] = [cookies.team, TRADING_MODE, created_at, since]",
      "var slot = 5",
    ),
    replace: L(
      "var where = `\"to\" = ?1 AND \"mode\" = ?2 AND \"synced_at\" > ?3`",
      "var binds: any[] = [cookies.team, TRADING_MODE, since]",
      "var slot = 4",
    ),
  },
  {
    id: "fix-trading-delta-cursor", worker: "trading",
    find: L(
      "var results: any[] = []",
      "try{",
      "var q = await env.DB.prepare(",
      "`SELECT * FROM items WHERE ${where} ORDER BY created_at DESC LIMIT ${limit}`",
      ").bind(...binds).all()",
      "results = q.results ? q.results : []",
      "}catch(err){",
    ),
    replace: L(
      "var results: any[] = []",
      "var cursor = Math.max(0, now - 1000)",
      "try{",
      "var q = await env.DB.prepare(",
      "`SELECT * FROM items WHERE ${where} ORDER BY \"synced_at\" ASC, \"id\" ASC LIMIT ${limit}`",
      ").bind(...binds).all()",
      "results = q.results ? q.results : []",
      "if(results.length >= limit){",
      "var lastRow = results[results.length - 1]",
      "var lastStamp = Number(lastRow.synced_at)",
      "var qTail = await env.DB.prepare(",
      "`SELECT * FROM items WHERE ${where} AND \"synced_at\" = ?${slot} AND \"id\" > ?${slot + 1} ORDER BY \"id\" ASC`",
      ").bind(...binds, lastStamp, lastRow.id).all()",
      "results = results.concat(qTail.results ? qTail.results : [])",
      "cursor = Math.min(cursor, lastStamp)",
      "}",
      "}catch(err){",
    ),
  },
  {
    id: "fix-trading-delta-cursor", worker: "trading",
    find: L("1초 겹치게 두어 중복 수신(무해, digest 로 스킵됨)을 택합니다.", "cursor: Math.max(0, now - 1000),"),
    replace: L("1초 겹치게 두어 중복 수신(무해, digest 로 스킵됨)을 택합니다.", "cursor: cursor,"),
  },
  //    POST: synced_at 열을 같이 쓰고, 그 값(쓰기 바로 앞의 서버 시각)은 반복이 끝난 뒤에 묶는다 — 문장은 [prepare, 인자] 로 모아 둔다
  {
    id: "fix-trading-delta-cursor", worker: "trading",
    find: L("statements.push(", "env.DB.prepare(`", "INSERT INTO items ("),
    replace: L("statements.push([", "env.DB.prepare(`", "INSERT INTO items ("),
  },
  {
    id: "fix-trading-delta-cursor", worker: "trading",
    find: L("\"mode\", \"digest\", \"data\", \"created_at\", \"updated_at\"", ") VALUES (", "?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13", ")"),
    replace: L("\"mode\", \"digest\", \"data\", \"created_at\", \"updated_at\", \"synced_at\"", ") VALUES (", "?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14", ")"),
  },
  {
    id: "fix-trading-delta-cursor", worker: "trading",
    find: "\"updated_at\" = EXCLUDED.\"updated_at\"",
    replace: L("\"updated_at\" = EXCLUDED.\"updated_at\",", "\"synced_at\"  = EXCLUDED.\"synced_at\""),
  },
  {
    id: "fix-trading-delta-cursor", worker: "trading",
    find: L("`).bind(", "id, docType, flagVal,", "cookies.address, cookies.team,", "cookies.cc, bccVal, refVal,", "TRADING_MODE, digestVal, blob,", "createdAt, updatedAt", ")", ")"),
    replace: L("`), [", "id, docType, flagVal,", "cookies.address, cookies.team,", "cookies.cc, bccVal, refVal,", "TRADING_MODE, digestVal, blob,", "createdAt, updatedAt", "]", "])"),
  },
  {
    id: "fix-trading-delta-cursor", worker: "trading",
    find: "if(statements.length){",
    replace: L(
      "var syncedAt = Date.now()",
      "statements = statements.map(function(p){ return p[0].bind(...p[1], syncedAt) })",
      "if(statements.length){",
    ),
  },

  // ── resolveTeam: 요청 팀 != 확인된 팀이면 본인 팀으로 강등 ──
  {
    id: "fix-trading-team-mismatch", worker: "trading",
    find: L(
      "console.log(`[TRADING] 🚫 (cache) 요청 팀 '${to}' != 확인된 팀 '${cached.team}'. 본인 팀으로 강등합니다.`)",
      "return { team: own, role: 'owner' }",
    ),
    replace: L(
      "if(cached.team == own){",
      "return { team: own, role: 'owner' }",
      "}",
      "console.log(`[TRADING] (cache) 요청 팀 '${to}' != 확인된 팀 '${cached.team}'. commerce 에 다시 묻습니다.`)",
    ),
  },
  {
    id: "fix-trading-team-mismatch", worker: "trading",
    find: L(
      "console.log(`[TRADING] 🚫 요청 팀 '${to}' != 확인된 팀 '${resolvedTeam}'. 본인 팀으로 강등합니다.`)",
      "return { team: own, role: 'owner' }",
    ),
    replace: L(
      "if(resolvedTeam == own){",
      "return { team: own, role: 'owner' }",
      "}",
      "console.log(`[TRADING] 🚫 요청 팀 '${to}' != 확인된 팀 '${resolvedTeam}'. 거부합니다.`)",
      "return { team: '', role: 'denied' }",
    ),
  },

  // ── findRelayValue 의 for-in 순서가 앱(키 정렬)과 다름 ──
  {
    id: "fix-trading-relay-order", worker: "trading",
    find: TOP,
    replace: L(
      TOP,
      "function compareCodePoints(a, b){",
      "var x = Array.from(a)",
      "var y = Array.from(b)",
      "for(var i = 0; i < x.length && i < y.length; i++){",
      "var d = x[i].codePointAt(0) - y[i].codePointAt(0)",
      "if(d){ return d }",
      "}",
      "return x.length - y.length",
      "}",
    ),
  },
  {
    id: "fix-trading-relay-order", worker: "trading",
    find: L("for(var k in data){", "if(!Object.prototype.hasOwnProperty.call(data, k)){ continue }"),
    replace: L("var relayKeys = Object.keys(data).sort(compareCodePoints)", "for(var rk = 0; rk < relayKeys.length; rk++){", "var k = relayKeys[rk]"),
  },
];
