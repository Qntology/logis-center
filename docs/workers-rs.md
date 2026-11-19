# Cloudflare Workers → workers-rs (2026-11-18 · 버그 수정 2026-11-19)

`workers/proxy`(proxy-commerce-logis-center) · `workers/apac`(commerce-apac-logis-center)의 TypeScript 를 Rust(workers-rs 0.8.7, wasm32)로
(2026-11-19 에 별도 저장소였던 commerce-logis-center · trading-logis-center 도 `workers/commerce` · `workers/trading` 으로 같은 방법으로)
다시 썼습니다. 목표는 **원본과 같은 동작**입니다. 같은 SQL, 같은 프롬프트 글, 같은 JSON(키 순서 · 수 표기), 같은 예외 경계를 지키고,
원본의 버그까지 그대로 옮겨 같음을 확인한 뒤(커밋 2e82ecbb), 버그를 고쳤습니다(아래 "고친 버그"). 원본은 커밋 `801b0ce0` 의
`workers/*/src/index.ts` 입니다.

## 구조

```
workers/                    자기 Cargo 작업 공간 (wasm32, 루트 작업 공간의 exclude)
├─ Cargo.toml  .cargo/      target-dir · wasm 의 getrandom 백엔드 (루트의 target-cpu=native 를 덮음)
├─ PORTING.md               옮기는 규칙 (JS → Rust 대응표)
├─ migrations/              D1 스키마 한 벌 (두 워커가 같이 씀, 파일 이름은 원본 그대로 — wrangler 가 이름으로 적용 이력을 본다)
├─ common/                  logis-worker-common
│  ├─ jsv.rs                JS 값 모형 V: 참조 공유 · undefined/null · 던지는 속성 읽기(엄격 모드) · JS 키 순서 · ryu-js 수 표기 · JSON
│  ├─ js.rs                 D1 · Workers AI · Vectorize · fetch 를 JS 그대로 부름 (wasm 전용)
│  ├─ gzip.rs               pako ungzip/gzip — D1 숫자 배열 입력에서 pako 가 던지는 자리까지 재현
│  ├─ date.rs · util.rs · dirty_json.rs · tables.rs · platform.rs
├─ commerce/                commerce-logis-center: 세션(S3) · 페이지 · 멤버 · 작업 대기열 API (원본 별도 저장소 fd1e991, 아래 "commerce")
├─ trading/                 trading-logis-center: 무역 문서 동기화 API (원본 65684e6, 새 D1 · migrations/ 는 이 워커 것)
├─ apac/                    cron 워커 (lib.rs 하나)
└─ proxy/                   fetch 워커
   ├─ handler.rs            처리기 뼈대: 머리 · 작업 반복 · 꼬리 · 배치 · 418
   ├─ image.rs              운송장 이미지 갈래
   ├─ scan.rs               화면 스캔 갈래 (페이지 판별 · 목록/상세 분석 · 아이템 반복 몰이)
   ├─ item1.rs ~ item4.rs   아이템 반복의 네 구역
   ├─ search.rs             자연어 검색 갈래
   ├─ ctx.rs                원본 처리기의 var 135개 (생성)
   ├─ templates.rs          원본의 템플릿 리터럴 175개 (생성) — 프롬프트 · SQL 은 손으로 다시 쓰지 않는다
   ├─ rules.rs · llm.rs     프롬프트 함수 · Relay 등 규칙, Gemini · DeepInfra 호출
   ├─ dom.rs + dom/         linkedom(htmlparser2 10.1.0 + entities 7) 과 같은 트리, selectors 크레이트로 CSS 선택
   └─ pug.rs                convertHtmlToCleanPug (LLM 에 보내는 페이지 글)
scripts/workers-gen/        templates.rs · ctx.rs · tables.rs 생성기
scripts/workers-diff/       차분 시험대 (아래)
```

- 원본의 `var` 는 함수 범위라 반복을 넘어 값이 남고 안쪽 반복이 바깥 셈 변수를 같이 씁니다. 그래서 처리기의 `var` 를 모두 `Ctx` 의 칸으로 두고,
  구역 함수가 `&mut Ctx` 를 나눠 씁니다. 구역 밖 반복의 `continue` · `break` 는 `Flow` 로 알립니다.
- workers-rs 0.9 는 `cloudflare:workers` 의 `tracing.getActiveSpan()` 을 무조건 부르는데, 지금 로컬 런타임(Miniflare/workerd 2026-07-30)에
  그 함수가 없어 모든 처리기가 예외로 끝났습니다. 0.8.7 로 고정했습니다(`worker-build@0.8.7` 도 같이).
- Vectorize 는 workers-rs 에 감싼 형이 없어 바인딩 객체의 `upsert` · `query` 를 JS 그대로 부릅니다(D1 · AI 도 같은 방식).

## 검증

| 무엇 | 결과 |
| --- | --- |
| 차분 시험 apac (`scripts/workers-diff`: `node apac.mjs`) | 고친 원본과 5/5 동일 — 바깥 요청(gzip 을 푼 본문) · D1 끝 상태 · 예외 |
| 차분 시험 proxy (`node proxy.mjs`) | 고친 원본과 44/44 동일 — 응답 · LLM 요청 본문 전부 · Workers AI · Vectorize 호출 · D1 11개 표 끝 상태(`console` 오류 기록 포함) |
| 고치기 전 (2e82ecbb) | 고치지 않은 원본과 apac 5/5 · proxy 23/23 동일 (`ORIGINAL=raw node proxy.mjs` 로 다시 볼 수 있음) |
| linkedom 대조 (`workers/proxy/tests/dom_vectors.rs`) | Pug 452/452 바이트 동일, 선택자 31,027/31,027 (SyntheticData 화면 96 · 경계 56 · 무작위 300). 따로 무작위 6,000 문서도 100 % |
| pako 대조 (`workers/common/tests/pako_array.rs`) | 430 경우 (93 던짐) 모두 같음 |
| `cargo test`(common · proxy), `cargo check --target wasm32-unknown-unknown` | 통과, 경고 없음 |
| 차분 시험 commerce (`node commerce.mjs`) | 고치기 전 원본과 28/28 (370d67a3), 고친 원본과 50/50 (fix-member-per-page 를 빼면 perpage_* 넷만 DIFF) — 응답(상태 · 머리글 · Set-Cookie · 본문) · S3 요청(서명은 aws4fetch 로 다시 계산해 확인) · D1 · S3 끝 상태 |
| 차분 시험 trading (`node trading.mjs`) | 고치기 전 원본과 41/41 (370d67a3), 고친 원본과 58/58 |
| `wrangler deploy --dry-run` (두 워커) | 빌드 · 바인딩 확인 통과. proxy 2,547 KiB (gzip 889 KiB), apac 419 KiB (gzip 166 KiB) |

차분 시험대는 원본(git 이력에서 꺼낸 TS 를 esbuild 로 묶음)과 Rust 빌드를 Miniflare 에서 같은 조건으로 돌립니다:
같은 D1 행, 같은 가상 시계(setTimeout 이 시계를 밀고 바로 실행), 같은 씨앗의 `Math.random`, 같은 가짜 LLM 응답(프롬프트로 어느 호출인지 가림),
같은 가짜 Workers AI · Vectorize(진입 모듈이 `env` 를 Proxy 로 감쌈). 시나리오는 이미지 · 검색(DeepInfra · Gemini · 망가진 JSON · 단일 문맥) ·
스캔(SyntheticData 의 실제 목록 · 상세 화면) · 오류(빈 본문 · 작업 없음 · 팀 없음 · rpm 소진 · 읽을 수 없는 gzip · 잘못된 URL)입니다.
`hashId()` 의 무작위 id(원본은 `ethers.Wallet.createRandom`)는 한쪽에만 나오는 주소를 나온 순서대로 자리표로 바꿔 견줍니다.
시간대는 운영과 같은 UTC 로 맞춥니다(Windows 의 workerd 는 `TZ` 를 따르지 않아 `prelude.mjs` 가 `Date` 를 UTC 로 감쌈).

"고친 원본"은 원본 TS 에 `fixes.mjs` · `fixes/*.mjs` 의 글 패치(수정마다 id, 공백을 무시하고 정해진 횟수만큼 맞아야 적용)를 덧댄 것입니다.
D1 의 undefined 바인딩 수정은 진입 모듈의 env 래퍼가 원본 쪽에도 똑같이 적용합니다. 수정마다 그것을 드러내는 시나리오가
`scenarios/*.mjs` 에 있고, 고치지 않은 원본(`ORIGINAL=raw`)과는 달라야 합니다.

```
cd scripts/workers-diff && npm install
(cd ../../workers/apac && worker-build --release) && (cd ../../workers/proxy && worker-build --release)
node apac.mjs && node proxy.mjs          # ONLY=scan_goods_list node proxy.mjs 로 하나만
```

## 고친 버그

| id | 원본 자리 | 무엇이 틀렸나 → 고침 |
| --- | --- | --- |
| fix-pako-array | 공용 | D1 은 BLOB 을 숫자 배열로 준다. pako 가 배열을 그대로 읽다 저장 블록(짧거나 잘 줄지 않는 글)에서 `input.subarray` 로 던져 작업을 통째로 버렸다 → 바이트로 바꿔 푼다 |
| fix-bind-undefined | 공용 | D1 은 undefined 바인딩을 `D1_TYPE_ERROR` 로 거절한다(배치가 원자적이라 그 지역 문장을 모두 잃음) → null 로 묶는다 |
| fix-isdiff-buffer | 공용 | 키 `data` 에서 `Buffer.isBuffer` — `nodejs_compat` 이 없어 ReferenceError → 다른 키처럼 견준다 |
| fix-clean-number | 1938 | `/./gi` 가 점이 있으면 모든 글자를 지웠다 → 점만 |
| fix-apac-lock-first | apac 158 | lock 행(늘 맨 앞)을 `ungzip(null)` 이 먼저 만나 lock 동안 매 회차가 예외 → 풀기 전에 lock 을 본다 |
| fix-apac-delete-quote | apac 318 | `id = "…"`(큰따옴표는 SQLite 에서 먼저 열 이름) → 바인딩 |
| fix-apac-gemini-quota | apac 421 | 쿼터는 2.5 · 2.0 flash-lite 키, 고르기는 flash-lite-latest 라 Gemini 가 한 번도 안 골라짐 → 쿼터 키를 맞춤 |
| fix-schema-sales-updated-at | 2732 | sales 표에 없는 `updated_at` 을 써서 배치 실패(418) → 빼고 status · tracking 만 |
| fix-schema-event-phone | 4980 · 5453 | event 표에 없는 `number` 열 → 같은 자리의 `phone` (값은 phone, 없으면 number) |
| fix-image-item-reset · fix-image-gemini-config · fix-image-order-slot · fix-image-items-bind · fix-image-embeddings | 2568 · 2578 · 2738 · 2927 · 2863 | 앞 작업의 item 이 남음, Gemini 에 config null(요청 뒤 던짐), pages[cc].order 칸 없음, items 삽입이 `item.*` 대신 `task.*`(digest · ref undefined), 죽은 임베딩 갈래 |
| fix-search-reset · fix-search-paragraphs-array · fix-search-condition-guard · fix-search-matches-length · fix-search-delete-index · fix-search-created-at · fix-search-primitive-results | 5706 · 5765 · 6020 · 6088 · 6130 · 6161 · 5784 | 첫 문맥의 queryVector 를 모든 문맥이 씀 등 지워지지 않는 var, 배열 아닌 paragraphs, 조건 없는 문맥, matches 없음, `results[i]`(→ `results[r]`), `indexOf(created_at)`(수 → `'"created_at"'`), 글 결과에 속성 쓰기 |
| fix-scan-reset · fix-scan-selectors-node · fix-scan-null-element · fix-detail-null | 3070 · 3335 · 3348 · 3122 · 3376 · 3386 · 3340 | 앞 페이지의 page · before · _page 가 남음, 풀지 않은 BLOB 의 `.node`, null 원소, `detail = null` 뒤 `detail.page` |
| fix-item-selectors-copy · fix-items-writeback · fix-item-semantic-reset · fix-goods-unshift · fix-condition-indexof · fix-relate-err-type · fix-tirm | 3769 · 3791 · 3984 · 3822 · 4199 · 4543 · 4047 | 늘 거짓인 조건(새 아이템은 선택자 없음), 병합 뒤 items[i] 와 갈라짐, 지워지지 않는 semantic, 글 goods 에 unshift, 글 · 배열 아닌 condition, 앞 회차의 type, `.tirm` 오타 |
| fix-relate-merge-reset · fix-relate-data-reset · fix-relate-semantic-reset · fix-embeddings-continue · fix-edge-data-ungzip · fix-edge-slots · fix-item4-embeddings-reset | 4594 · 4636 · 4708 · 4782 · 4797 · 4660 · 5190 | 앞 행의 merge · data 가 남음, 뒤집힌 `if(embeddings) continue`, 이미 객체인 data 를 ungzip(스캔 전체 중단), 없는 pages 칸 |
| **fix-edge-draft** | **5087** | 관계 행이 없을 때 앞 회차의 `edgeType` 으로 `env["commerce_logis_<zone>_undefined"]` 를 찾아 던져 **모든 스캔이 첫 아이템에서 멈췄다** → 관계 질의(`query[1] ?? query[0]`)의 표 · 값으로 초안 대상(`hashId(type + item.id)`)을 만든다 |
| fix-undefined-binds | 5062 · 5102 | 다른 행의 id(`row.id` · `item.id`)를 묶음 → 확인한 행(`edge.id`) |
| fix-parse-body | 1984 | 인자를 가리는 `var body = ''` 와 정의되지 않은 `selectors` (부르는 곳 없음) |
| fix-sql-params | 전체 | SQL 에 값을 이어 붙였다(작은따옴표가 든 값이면 깨짐) → 모두 `?n` 바인딩. 표 이름은 알려진 표만, 조건 조각(parseCondition)은 유한한 수만, ORDER BY 열은 그 표의 열만 |

판단이 들어간 것: fix-edge-draft 의 초안 유형은 관계 키(`query[0].type`)입니다(`relate.type` 이 아님 — 찾지 못한 질의와 같은 유형으로 다시 찾게).
fix-schema-* 는 마이그레이션 대신 코드를 스키마에 맞췄습니다 — `predeploy` 는 `DB`(logis_center)에만 마이그레이션을 적용하고, 이 문장들이 쓰는
지역 DB(`commerce_logis_apac1_*`)에는 적용하지 않기 때문입니다. 운영 지역 DB 에 그 열이 따로 만들어져 있다면 되돌릴 자리입니다.
닿는 시나리오가 없는 수정(같은 요청 안에서 작업은 `LIMIT 1` 이라 "앞 작업" 누수는 재현되지 않음, `CLIENT_EMBEDDING` 이 참이라 죽은 갈래)은 코드 검토로만 확인했습니다.

## trading (trading-logis-center 65684e6) 고친 버그

`workers/trading` 은 원본과 41/41 같음을 확인한 뒤(370d67a3) 아래를 고쳤습니다. Rust 의 `// 고침 (fix-trading-…)` 과 원본에 덧대는
`scripts/workers-diff/fixes/trading.mjs` 가 1:1 입니다. `node trading.mjs` 는 고친 원본 · `ORIGINAL=raw` 는 고치지 않은 원본 ·
`OMIT_FIX=<id>` 는 그 수정 하나만 뺀 고친 원본과 견줍니다(수정은 서로 같은 글을 찾지 않아 하나씩 뺄 수 있음 — DIFF 가 나는 시나리오가
곧 그 수정이 바꾸는 시나리오). 실제 소비자는 데스크톱 앱(`web/desktop/src/modes/trading.ts` → `proxy_fetch`)뿐이고, 그 GET · POST
꼴 그대로의 시나리오(`app_*`)가 모두 받아들여짐을 확인합니다.

| id | 원본 자리 | 무엇이 틀렸나 → 고침 |
| --- | --- | --- |
| fix-trading-cookie-identity | 484 · 629 · 754 · 991 | Cookie 의 모든 이름을 cookies 에 담아, S3 세션이 실패해도 `address` · `team` 쿠키만으로 남의 팀을 읽고 쓰고 지움 → 머리글에서는 `hash` · `token` 만 |
| fix-trading-token-required | 548 | S3 token 메타도 요청 token 도 없으면 `null == undefined` 로 hash 만으로 통과, commerce 에 `token=undefined` → 둘 다 있고 같아야 |
| fix-trading-setcookie-verified | 559 | 확인 안 된 hash · token 을 Set-Cookie 로 되돌려 `?hash=x;%20Domain=logis.center` 로 쿠키 속성 끼워 넣기 → 확인된 세션만 |
| fix-trading-token-echo | 모든 응답 | 모든 `session`(과 HTML)에 token — HttpOnly 로 감춘 비밀을 본문으로 → 본문에서 뺌 (쓰는 소비자 없음) |
| fix-trading-html-escape | 619~625 | `<pre>${JSON}</pre>` 반사 XSS (S3 가 200 을 주는 hash 뒤에 `#<script>` 등) → HTML 이스케이프 |
| fix-trading-cors | 605~609 | 아무 Origin 이나 `Allow-Credentials` 와 함께 반사 → `https://logis.center` · `commerce.` · `trading.` 만 (Origin 없음은 원본 그대로) |
| fix-trading-options-first | 605~613 | preflight 가 세션 뒤라 OPTIONS 마다 bootstrap S3 PUT → 세션보다 먼저 답함 |
| fix-trading-upsert-scope | 806~812 · 910~922 | `ON CONFLICT (id) DO UPDATE` 가 팀을 보지 않아 남의 행을 덮어쓰고 가져감 → 사전 조회를 팀 없이 해 다른 팀 id 는 거부 + `WHERE items."to" = EXCLUDED."to"` |
| fix-trading-preload-chunk | 809~812 | 1 + id 수 바인딩이 100 을 넘으면 D1 이 거절, 오류를 삼켜 SKIP GUARD 가 조용히 꺼짐 → 99건씩 |
| fix-trading-delta-cursor | 643~668 · 733 · 903 | 클라이언트 시각 `max(updated_at, created_at)` 을 서버 커서와 견주어, 커서보다 옛 시각으로 나중에 들어온 행(오프라인 편집 · 처음 올리는 옛 서류 · 시계 늦은 기기)을 다른 기기가 영영 못 받음, `created_at <` 상한이 시계 빠른 기기 서류를 흘림, LIMIT 에 잘려도 커서를 now 로 → 서버가 쓸 때 찍는 `synced_at`(migrations/0002)으로 고르고 `synced_at, id` 순으로 쪽을 나눔. 꽉 찬 쪽은 마지막 시각의 남은 행을 붙이고 커서를 그 시각으로 |
| fix-trading-id-case | 800 · 828 · 998 | id 를 trim · 소문자로 저장 — 앱(store.rs)은 id 를 그대로 써서 내려받으면 다른 문서(로컬 두 벌 · 묘비 · 봉투 반영 실패) → 보낸 그대로 (공백뿐인 id 는 거부) |
| fix-trading-nan-params | 639 · 661 · 858 · 866 | `since=` · `limit=` · `Infinity` 가 `LIMIT NaN` · NaN 바인딩으로 0건(커서도 없음), `created_at: "Infinity"` 가 NULL 로 저장 → 기본값 |
| fix-trading-team-mismatch | 412 · 452 | 요청 팀 != 확인된 팀이면 본인 팀으로 강등 — 다른 팀 멤버의 서류가 개인 팀에 저장(함수 주석이 막으려던 오염) → 확인된 팀이 본인 팀이면 본인 팀, 아니면 거부. 캐시가 다르면 다시 물음(팀을 옮긴 뒤 5분 강등 없음) |
| fix-trading-cf-optional | 464 | `request.cf` 가 없으면 try 밖 TypeError 로 처리기가 던짐 → `{}` |
| fix-trading-relay-order | 298 | `for-in` 순서(정수 키 먼저)가 앱 `find_relay_value`(키 정렬)와 달라 서버 ref ≠ 로컬 ref → 코드 포인트 순 |
| fix-trading-proto-keys | 476 · 496 · 716 · 797 · 806 · 962 | 평범한 객체 조회: id `constructor` 는 사전 조회에서 빠져 늘 다시 씀, `__proto__` 행이 프로토타입을 바꿈 → `Object.create(null)` · hasOwnProperty |
| fix-trading-double-decode | 654 · 1004 · 1008 | 이미 풀린 쿼리를 다시 `decodeURIComponent` — `%` 가 든 id 는 URIError(418), `%41` 은 `A` 로 → 한 번만 |
| fix-trading-inflate-grow | Rust 판만 (jsrt.rs) | pako 흉내 inflate 가 출력 칸(입력 × 4 + 1 KiB)이 차면 'buffer error' — 4배 넘게 줄어든 gzip 본문(앱의 200건 push)을 못 풀어 POST 가 빈 목록 → 칸을 늘려 끝까지 |

클라이언트 쪽 짝: `web/desktop/src/modes/trading.ts` 의 push 커서가 `accepted + skipped` 만 세어, 거부된 행 하나에 커서가 묶였습니다
(BUG-TR-3) → `!res.error && accepted + skipped + rejected >= batch.length`. 같은 시각이 200건을 넘으면 경계 행을 못 올리는
BUG-TR-2(`it.fails`)는 커서 꼴(시각 + id)을 바꿔야 해 그대로 둡니다.

고치지 않은 것: `stableStringify` 는 중첩 객체의 volatile 키도 이미 뺍니다(재귀가 매 단계 거름 — 시나리오 `kv`). 앱의 `digest` 는
다른 함수(`logis_schema::hash::digest`)라 서버 폴백 값과 원래 같지 않습니다(클라이언트 digest 가 1순위).

## commerce (commerce-logis-center fd1e991) 고친 버그

`workers/commerce` 는 원본(HEAD fd1e991 — 1594 줄의 짝 없는 `}else if` 한 곳만 고침, 하네스가 raw 에서도 고침)을 구역별로 옮긴 것입니다:
`session.rs`(341~965) · `resolve.rs`(966~1593) · `put_delete.rs`(1594~1997) · `post.rs`(1998~2454) · `get.rs`(2455~2869), 고친 동작의 도우미는
`fixes.rs`. 먼저 원본 버그까지 그대로 옮겨 28/28 같음을 확인했고(370d67a3), 아래를 고쳤습니다. 원본 쪽 패치는
`scripts/workers-diff/fixes/commerce.mjs` + `commerce_patches.txt`(긴 JS 블록), `COMMERCE_SKIP=<id>` 는 그 수정 하나만 뺍니다.

| id | 원본 자리 | 무엇이 틀렸나 → 고침 |
| --- | --- | --- |
| fix-cookie-whitelist | 421~428 | Cookie 머리글을 통째로 믿어 `address` · `team` · `sender` · `balance` · `cc` 를 위조할 수 있었음(지금은 `zoneRegion` 이 비어 우연히 418, 그 전에 `team` 이 SQL 글로 실행) → `hash` · `token` 만 |
| fix-session-token | 607 | S3 hash 객체에 token 메타가 없으면 `null == undefined` 로 hash 만으로 로그인 → 토큰이 있어야 |
| fix-sql-bind | 곳곳 | 쿠키 · DB 에서 온 글을 SQL 에 이어 붙임(2172 · 2187 은 큰따옴표) → 바인딩. 검증된 주소 · 수 · 서버 해시는 그대로 |
| fix-preflight-session | 922 | 쿠키 없는 preflight 마다 새 지갑 + S3 PUT → OPTIONS 는 세션 전에 답함 |
| fix-cors-methods | 969 | Allow-Methods 에 PUT · DELETE 없음 → 더함 |
| fix-delete-tasks-db | 1970 | `DELETE FROM tasks` 를 tasks 가 없는 items DB 에 큰따옴표로 → 지역 DB 에서 `ref` · 내 `from` · 내 팀 `to` 로 묶어서 |
| fix-task-column | 2397 · 2359 | `?error=` 재시도의 ON CONFLICT 가 없는 `"task"` 열 → 실제 열, 그리고 "이미 있음" 조기 반환을 건너뛰어 다시 넣을 수 있게 |
| fix-coupon-count | 1119 | `data.base.event.coupon`(없음) → `data.base.coupon.count` |
| fix-tracking-index | 2746 | 지난 반복의 `item.index`(또는 COUNT 행 → `"order" = undefined` 로 418) → `$item.index` |
| fix-draft-count | 2552 | `undefined++` 가 NaN(JSON null) → 0 부터 |
| fix-revoke-owner | 1738 · 1768 | `team.from` — `team` 은 팀 이름 글이라 "소유자는 못 내보냄" 검사가 늘 통과 → 팀 행(`admin`) 또는 users 의 팀 행에서 소유자 |
| fix-link-hoist | 666 | `link` 를 996 줄에서야 정해 `user.ref = hashId(…+undefined)` → 같은 방법으로 먼저 계산 |
| fix-zone-pool | 700~863 | 빈 자리(pool < 1000) 있는 지역이 없으면 `zoneRegion = ''` 로 사용자를 저장해 영영 `commerce_logis__items` → 만들지 않고 로그인 안 된 세션 |
| fix-region-fallback | 401 | wrangler 는 APAC DB 만 묶는데 미국 등은 `commerce_logis_enam` 이라 모든 요청 418 → 묶이지 않은 지역은 apac |
| fix-crons-query | 1414~1468 | `?crons=` JSON 글을 글자마다 돌고, 주소가 없으면 `WHERE  ORDER BY`, 지울 자리를 crons 순번으로 → 배열만 · 98개 · 바인딩 · id 로 지움 |
| fix-app-cloud-search | 2030 · 2113 | 앱의 `{query, vector}`(from=주소 · to=팀)가 글 갈래로 가 벡터가 버려짐, 검색 작업에 `contentType` 이 없어 proxy 가 `indexOf` 에서 버림 → 방향과 무관하게 질의 갈래, contentType 채움, 홈 페이지 |
| fix-app-extraction | 1998~ | 앱 추출 POST 는 `created_at` 이 없어 418 → 기본 now, 봉투 `{type, body, link…}` 를 풂: 이미지 = data URL 의 base64 + `image/*`, html = 스캔 갈래, 문서 = `{success:false, error}` (proxy 에 문서 해석기 없음) |
| fix-app-relay | (없던 길) | `/relay/{hash}` 가 없었음 → POST 는 `hash/{hash}` 의 토큰이 맞아야 S3 `relay/{hash}` 에 저장(64 KB 이하 JSON), GET 은 10분 안의 것. 앱(main.ts `sendSignalingMessage`)이 세션을 안 실어 보내던 것도 고침 |
| fix-app-invite | 1723~1889 | 앱의 초대 PUT(email · ref[] · id 없음 · `results[0].hook` 기대)을 받는 곳이 없었고, 권한 주기가 쓰는 행 꼴과 멤버 판정이 읽는 꼴이 달랐음 → 아래 |
| fix-member-per-page | 1108 · 1561~1577 · 2000 · 1894 · 2494 | 팀 행(늘 읽힘)이 `member = true` 라 팀원 모두가 모든 페이지의 멤버였고, member 는 GET 동기화만 막았음(POST · PUT 대화는 보지 않음) → 팀장 · 페이지 멤버 · 팀 공통 페이지만 멤버, `cookies.pages` 거름, POST · PUT 대화도 멤버만 (아래 "페이지별 멤버") |

멤버 초대(사용자 결정: 대기 초대 → 첫 로그인 때 확정):

- 초대 PUT: 검증된 세션 · `from` = 내 팀 · `to` = 나 · 내가 팀 소유자 · 이메일 꼴이면, 지역 DB `users` 에
  `id = hashId(team + email)` · `type 'invite'` · `from` 초대한 사람 · `to` 팀 · `cc = hashId(email)` · `data = gzip({email, refs, zone, flag, name})` 를
  upsert 하고 `{results:[{id, table:'users', type:'invite', hook}]}` 로 답합니다. `hook` 은 앱의 기존 기본값(`${hash}.logis.center@oauth.email`) 그대로입니다.
- 첫 로그인(사용자 행이 아직 없을 때): 그 이메일의 초대가 있으면 새 팀 대신 초대한 팀 · 그 지역으로 들어가고(팀 행 · pool 증가 없음),
  고른 페이지마다 멤버 행을 만든 뒤 초대 행을 지웁니다.
- 멤버 행(초대 · 권한 주기 · 내보내기가 같이 씀, 1567 의 판정이 읽는 꼴): `type 'member'` · `from` 멤버 주소 · `to` 팀 · `cc` 준 사람 ·
  `bcc` = 그 페이지 행의 `bcc`(`hashId(type + cc)`) · `ref` 페이지 id · `id = hashId(member + team + bcc)`.
- 다른 지역(logisRegion)의 초대는 찾지 않습니다.

페이지별 멤버(fix-member-per-page, 사용자 결정: "페이지별로 막는 규칙으로"). 원본은 팀에 속한 사용자(`users.to` = 팀)면 1108 줄에서 팀 행을 읽는
순간 `member = true` 였습니다(팀 행은 `?created_at=` 이 있는 요청마다 읽힘). 이제 판정은 1561 한 곳(Rust `resolve.rs`)입니다:

- 팀장: 팀 행의 `from`(`teamOwner()` — 1089 가 찾은 admin 이 있으면 그것, 없으면 users 의 팀 행을 읽음, fix-revoke-owner 와 같은 도우미).
  `cookies.address` 가 팀장이면 모든 페이지의 멤버. 팀 행을 못 읽은 요청(`created_at` 없음)에서도 팀장이면 멤버입니다(원본은 그때 아니었음).
- 페이지 멤버: 그 밖의 팀원은 멤버 행이 있어야 합니다 — `from` = 나 · `to` = 팀 · `bcc` = 이 페이지의 `cookies.bcc`(`hashId(page.type + cc)`, 상세면 대문자 cc),
  또는 `bcc` = `cookies.cc`(사이트 멤버 행: 그 사이트의 모든 페이지). 바인딩한 원래 두 질의 그대로이고, 페이지(`page`)를 못 찾은 요청이면 멤버가 아닙니다.
- 팀 공통 페이지: 요청 href 의 도메인 cc(`ccOfHost(url.host)`, 세션 505~517 과 같은 규칙)가 `homeCc` = `hashId('logis.center')` 이면 팀원 누구나 멤버.
  앱의 클라우드 검색(`href=https://commerce.logis.center/tracking`, `origin=https://commerce.logis.center`)과 추출(`href=https://console.logis.center/`,
  origin 없음 → `cookies.cc` 가 없어 `cookies.cc` 로는 못 가림)이 여기에 듭니다. `homeId`/`pageId` 로는 가릴 수 없습니다(홈 id 는 대문자 cc 로 만들어
  `pageId` 와 같아지지 않고, fix-app-cloud-search 의 `page = home` 은 POST 본문을 읽을 때에야 정해짐). 앱의 페이지 목록에 없어 초대로는 이 페이지의
  멤버 행이 생기지 않으므로, 막으면 초대받은 팀원은 앱을 쓸 수 없습니다.
- `cookies.pages`: 팀 `base.pages` 는 proxy 가 채우는 `{ [사이트 cc]: { [페이지 type]: { draft, count } } }`(페이지 id 가 아님). 팀장은 그대로,
  그 밖에는 그 사람의 멤버 행 `bcc` 로 거릅니다 — cc 가 `homeCc` 이거나 사이트 멤버 행(`bcc` = cc)이 있으면 그 cc 통째로, 아니면
  `bcc` = `hashId(type + cc)`(또는 대문자 cc)인 type 만, 남는 type 이 없는 cc 는 뺍니다(`fixes.rs::member_pages`). 개수(goods · order · …)는 팀 것 그대로.
- 쓰기: POST(화면 글 · 이미지 · `{query, vector}` · URL 목록 · 앱 추출 전부)와 PUT 대화는 `member` 여야 합니다. 아니면 200
  `{success:false, results:[], error:"not a member of this page", session}`(PUT 은 `{results, success, error, session}` 순서 — 초대 거절과 같은 꼴),
  Content-Type `application/json`. 초대 · 권한 주기는 팀장만, 거두기는 팀장 또는 본인 — 그대로입니다. 멤버가 아닌 JSON GET 에는 동기화(2494~)뿐 아니라
  1132 의 상세 아이템도 싣지 않습니다(페이지 · 팀 사용자 목록은 그대로).
- PUT 벡터(Vectorize upsert)는 막지 않습니다: 데스크톱 앱이 팀의 로컬 색인을 통째로 올리는 팀 단위 작업이고 어느 페이지에 딸린 것이 아니며,
  검증된 세션의 팀(`cookies.team`)이 있어야 하는 원래 조건으로 충분합니다. (벡터마다 `to` · `cc` 를 덮어쓸 수 있는 것은 원래 동작이고 고치지 않았습니다.)
- 결과: `users.to` = 팀이지만 팀장이 아니고 멤버 행이 없는 사용자 — 이 변경 전에 들어온 팀원, refs 없이 초대된 사람 — 은 이제 팀 공통 페이지만 봅니다.
  팀장이 권한 주기(PUT `id` = 팀 · `cc` = 팀장 · `ref` = 페이지들)로 페이지를 주어야 합니다. 의도한 결과입니다.
- 남은 것: `member` 가 참이면 원래의 동기화(2494~2849)는 페이지가 아니라 팀 · 사이트 단위 질의(같은 cc 의 아이템, 다른 cc 의 최근 아이템, 팀 초안)라,
  한 페이지의 멤버는 동기화로 팀의 다른 페이지 아이템도 받습니다. 앱 추출의 html 봉투는 `link` 의 가게 페이지로 작업을 넣지만 판정은 요청 href(console)로 합니다.

고치지 않고 적어 둔 것: `cookies.hash` 를 S3 경로에 검증 없이 씀, 홈 id 를 대문자 cc 로 만들어 `pageId` 와 같아지지 않음(클라우드 검색만 보정),
`?created_at=%20` 의 NaN 이 SQL 로(418), 1474~1558 의 페이지 트리는 결과를 버리는 죽은 코드, POST 가 sender · 잔액 없으면 빈 418,
DELETE 응답의 Content-Type 없음, POST 성공 응답의 `ref` 에 href.

하네스 한계: 한쪽에만 있는 40 · 64자리 16진 값은 자리표가 되므로, **고치지 않은 원본과 견줄 때** 해시 값만 바뀌는 수정(fix-link-hoist 의
`users.ref`)은 SAME 으로 보입니다. 고친 원본 vs Rust 비교는 같은 값을 그대로 두므로 영향이 없습니다.

## 알려진 차이 (재현할 수 없거나 관찰되지 않는 것)

- gzip 압축 바이트는 pako 와 다릅니다(miniz_oxide). 푼 글은 같고, 저장 · 비교는 늘 풀어서 하므로 동작은 같습니다.
- CSS 선택자: css-select 의 확장(`:contains` · `:checked` · `:has-text` 류, `[attr!=v]`)은 SyntaxError 가 됩니다(원본은 맞춤). LLM 이 낸 선택자에만 해당.
- `new Date("…")` 는 ISO 8601 과 `YYYY/MM/DD[ hh:mm[:ss]]` 만 읽습니다(V8 의 옛 해석기가 받는 그 밖의 꼴은 NaN). 프롬프트가 ISO 꼴을 요구합니다.
- BMP 밖의 이름 있는 엔티티는 linkedom 에서 짝 없는 서로게이트 두 글 노드가 되는데, Rust 문자열은 그것을 담지 못해 Pug 줄에 U+FFFD 로 나옵니다.
- 예외 메시지 몇 곳(우리가 직접 던지는 TypeError)은 V8 의 글과 조금 다릅니다. 로그와 `console` 표에만 남습니다(시험한 경로에서는 모두 같았음).
  pako 가 던지는 글(`buffer error` · `incorrect header check`)은 같게 맞췄습니다.
- `hashId()`(인자 없음)는 원본이 BIP-39 니모닉으로 지갑을 만들어 그 개인 키를 쓰고, Rust 는 무작위 32바이트를 씁니다. 둘 다 무작위 id 입니다.

## 배포

`wrangler.json` 의 바인딩 · 이름 · 크론은 그대로이고, `main` 이 `build/index.js`, `build.command` 가
`cargo install -q worker-build@0.8.7 --locked && worker-build --release` 입니다. 배포 PC 에는 Rust 와 `wasm32-unknown-unknown` 표적이 있어야 합니다.
`npm run deploy`(predeploy 가 `migrations/` 를 DB 에 적용).

- commerce: 정적 자산(셸 HTML 이 읽는 `static/js/content.js` 등)은 `web/commerce/` 에서 올라갑니다(`assets.directory`). 비밀 값
  `aws_access_key_id` · `aws_secret_access_key` · `aws_region` · `aws_bucket` 은 `wrangler secret put` 으로(원본과 같음).
- trading: 아직 배포된 적 없는 워커입니다. `wrangler d1 create trading_logis_center` 로 만든 id 를 `wrangler.json` 의 자리표시
  `00000000-…` 에 넣고, 위 aws 비밀 네 개를 넣은 뒤 `wrangler d1 migrations apply DB --remote`(0001 · 0002) → `wrangler deploy`.
  경로는 `trading.logis.center`(custom domain).
