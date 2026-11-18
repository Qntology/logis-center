# TypeScript 워커 → workers-rs 옮기는 규칙

원본은 커밋 `801b0ce0` 의 `workers/proxy/src/index.ts` · `workers/apac/src/index.ts` 입니다(`git show 801b0ce0:workers/proxy/src/index.ts`).
목표는 **같은 동작**입니다: 같은 SQL · 같은 프롬프트 글 · 같은 JSON(키 순서 · 수 표기) · 같은 예외 경계. 원본의 버그도 고치지 않고 옮기며
(`// 원본 그대로: …` 주석), 고칠 것은 옮긴 뒤 따로 합니다. 같은지는 `scripts/workers-diff`(Miniflare 로 두 판을 같은 D1 · 같은 가짜 응답으로
돌려 바깥 요청 · D1 끝 상태를 견줌)가 판정합니다.

## 값: `logis_worker_common::V`

| JS | Rust |
| --- | --- |
| `undefined` · `null` | `V::Undef` · `V::Null` |
| `{ a: 1, b: x }` · `[a, b]` | `obj! { "a" => 1, "b" => x.clone() }` · `arr![a, b]` 또는 `V::from(vec![...])` |
| `a.b` (a 가 undefined 면 던짐) | `a.g("b")?` · 계산된 키 `a.gk(&k)?` · 첨자 `a.gi(i)?` |
| 던지지 않는 읽기(확실히 객체일 때) | `a.get("b")` · `a.at(i)` |
| `a.b = v` · `a[k] = v` | `a.s_("b", v)?` · `a.sk(&k, v)?` (a 가 undefined 면 던짐) |
| `delete a.b` | `a.del("b")?` |
| `arr.push(x)` | `arr.push(x)?` (배열이 아니면 TypeError) |
| `x.length` | `x.length()?` (V) · 배열 길이만 쓰면 `x.len()` |
| `Object.keys(x)` · `for (k in x)` | `x.keys_checked()?` · `x.keys()` (JS 순서: 첨자 꼴 키 먼저) |
| `x.hasOwnProperty(k)` | `x.has(k)` |
| `if (x)` · `!x` | `x.truthy()` · `!x.truthy()` |
| `==` · `===` · `<` | `loose_eq(&a, &b)` · `strict_eq(&a, &b)` · `less_than(&a, &b)` (NaN 이면 None) |
| `a + b` | `add(&a, &b)` (문자열이 하나라도 있으면 이어 붙임) |
| `x += 1` · `x++` · `x--` | 수: `V::Num(x.to_num() + 1.0)` (문자열일 수 있으면 `add`) |
| `typeof x` | `x.type_of()` |
| `String(x)` · `${x}` | `x.to_str()` |
| `Number(x)` · `parseInt` · `parseFloat` | `x.to_num()` · `jsv::parse_int` · `jsv::parse_float` |
| `JSON.stringify(x)` · `JSON.parse(s)` | `stringify_str(&x)`(undefined → "undefined") / `stringify_checked` · `parse(&v)?` |
| `safeClone(x)` | `x.deep_clone()` |
| `{ ...a }` | 새 `V::new_obj()` 에 `a.keys()` 순서로 `put` |
| `[...a, ...b]` | `V::from(a.items().into_iter().chain(b.items()).collect::<Vec<_>>())` |
| `s.indexOf` · `includes` · `substring` · `slice` · `split` · `trim` | `jsv::index_of` · `includes` · `substring` · `slice` · `split` · `js_trim` (UTF-16 위치) |
| `s.toUpperCase()` · `toLowerCase()` | `jsv::to_upper` · `to_lower` (문자열이 아니면 원본처럼 던질 것) |
| `s.replace("a", "b")` | **첫 번째만** 바뀐다: `s.replacen("a", "b", 1)` |
| `s.replace(/re/g, …)` | `regex` (미리보기 등이 있으면 `fancy_regex`) — JS 정규식과 뜻이 같은지 확인 |
| `new Date(v).getTime()` · `toISOString()` | `date::get_time(&v)` · `date::to_iso_string(t)?` |
| `Date.now()` · `Math.random()` | `platform::now()` · `platform::random()` (`Math.floor(Math.random()*n)` = `util::random_below(n)`) |
| `console.log('x', y)` | `console_log!("x", y)` — 원본의 꼬리표 그대로 |
| `new URL(s)` | `url::Url::parse(&s)` (실패 → `Throw::type_error("Invalid URL")`), `pathname` = `.path()` |

## 변수: `Ctx`

원본 fetch 처리기의 `var` 135개는 모두 `crate::ctx::Ctx` 의 칸입니다(`$x` → `s_x`, `type` → `type_`, `match` → `match_`).
JS 의 `var` 는 함수 범위라 반복을 넘어 값이 남고 안쪽 반복이 바깥 셈 변수(`t` · `i` …)를 같이 씁니다 — 그래서 **반복의 셈 변수도 `ctx` 칸**으로
씁니다(`ctx.i = V::Num(0.0); while ctx.i.to_num() < … { …; ctx.i = V::Num(ctx.i.to_num() + 1.0) }`). `let` · `const` 만 지역 변수.
`var x` 를 다시 선언해도 값이 지워지지 않습니다(초기값이 있을 때만 대입).

## 템플릿 리터럴: `tpl(T<줄>, &[식…])`

원본의 모든 `` `…${x}…` `` 는 `crate::templates` 에 줄 번호 이름(`T3181`, 한 줄에 둘 이상이면 `T3741_1` …)으로 있습니다. 조각 사이에 들어갈 식의
원문이 상수 위 주석에 있습니다. `tpl(T3181, &[ctx.language.clone(), list2json(&ctx.language)])` 처럼 같은 순서로 넘깁니다.
`${JSON.stringify(x)}` 는 `V::from(stringify_str(&x))`. 식이 없는 템플릿(SQL)은 `T6416[0]`. **프롬프트 · SQL 글을 손으로 다시 쓰지 마세요.**

## 예외 · 흐름

- `try { … } catch (err) { console.log('x', err) }` → `let r: R<()> = async { … Ok(()) }.await; if let Err(err) = r { console_log!("x", err); }`
  (동기면 클로저). try 밖으로 새는 예외는 `?`.
- 구역 밖 반복의 `continue` · `break` 는 `return Ok(Flow::Continue)` · `Ok(Flow::Break)`. 끝까지 돌면 `Ok(Flow::Next)`.
- 구역 안 반복의 `continue` 는 JS `for` 의 증가식이 돈다는 것을 잊지 말 것(Rust `while` 에서 `continue` 전에 셈을 올린다).

## 바인딩

- D1: `env.prepare(db, sql)?` → `js::bind(&stmt, &[…])?` → `js::all(&stmt).await?`(`{ results, success, meta }`) · `js::run(&stmt).await?`.
  `var { results } = await env[db].prepare(sql).all()` 는 `ctx.results = handler::select(env, db, &sql).await?`.
  `statements[key].push(env[db].prepare(sql).bind(…))` 는 `ctx.statements.get(key).push(handler::prepare_bind(env, db, sql, &[…])?)?`.
- Workers AI: `env.ai_run(model, &input).await?` · Vectorize: `env.vectorize_upsert(name, &vectors).await?` · `env.vectorize_query(name, &vector, &options).await?`.
- LLM: `crate::llm::deepinfra(&key, &model, &system, &user, &config, &inline_data).await?` · `crate::llm::gemini(…)` (빠진 인자는 `V::Undef`, `null` 은 `V::Null`).
- gzip: `gzip::ungzip_json(&row.g("data")?)?` · `gzip::gzip_json(&x)?`(D1 에 넣는 ArrayBuffer).
- 표: `tables::lookup(LOGIS_REGION, &flag)` · `LANGUAGE_CODE` · `CENTER_REGION`.
- 규칙 · 프롬프트: `crate::rules::{image2json, list2json, item2json, related, relay, parse_status, paragraph2propertys, parse_condition, …}`,
  `logis_worker_common::util::{hash_id, digest, crc32, merge_node, is_diff, clean_number, normalize_numeric_homoglyphs, rows_trim, …}`.
- DOM: `crate::dom::Document::parse(&html)` → `query_selector_all(sel)?` · `query_selector(sel)?` · `text_content(id)` · `get_attribute(id, name)`.
  HTML → Pug 는 `crate::pug::convert_html_to_clean_pug(&body)`.

## 검사

`cargo check -p logis-worker-proxy --target wasm32-unknown-unknown` 이 경고 없이 통과해야 합니다. 차분 시험은 Miniflare 시험대에서.
