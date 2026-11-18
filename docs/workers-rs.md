# Cloudflare Workers → workers-rs (2026-11-18)

`workers/proxy`(proxy-commerce-logis-center) · `workers/apac`(commerce-apac-logis-center)의 TypeScript 를 Rust(workers-rs 0.8.7, wasm32)로
다시 썼습니다. 목표는 **원본과 같은 동작**입니다. 같은 SQL, 같은 프롬프트 글, 같은 JSON(키 순서 · 수 표기), 같은 예외 경계를 지키고,
원본의 버그도 고치지 않고 옮겼습니다(아래 "원본의 버그"). 원본은 커밋 `801b0ce0` 의 `workers/*/src/index.ts` 입니다.

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
| 차분 시험 apac (`scripts/workers-diff`: `node apac.mjs`) | 5/5 동일 — 바깥 요청(gzip 을 푼 본문) · D1 끝 상태 · 예외 |
| 차분 시험 proxy (`node proxy.mjs`) | 23/23 동일 — 응답 · LLM 요청 본문 전부 · Workers AI · Vectorize 호출 · D1 11개 표 끝 상태(`console` 오류 기록 포함) |
| linkedom 대조 (`workers/proxy/tests/dom_vectors.rs`) | Pug 452/452 바이트 동일, 선택자 31,027/31,027 (SyntheticData 화면 96 · 경계 56 · 무작위 300). 따로 무작위 6,000 문서도 100 % |
| pako 대조 (`workers/common/tests/pako_array.rs`) | 430 경우 (93 던짐) 모두 같음 |
| `cargo test`(common · proxy), `cargo check --target wasm32-unknown-unknown` | 통과, 경고 없음 |
| `wrangler deploy --dry-run` (두 워커) | 빌드 · 바인딩 확인 통과. proxy 2,547 KiB (gzip 889 KiB), apac 419 KiB (gzip 166 KiB) |

차분 시험대는 원본(git 이력에서 꺼낸 TS 를 esbuild 로 묶음)과 Rust 빌드를 Miniflare 에서 같은 조건으로 돌립니다:
같은 D1 행, 같은 가상 시계(setTimeout 이 시계를 밀고 바로 실행), 같은 씨앗의 `Math.random`, 같은 가짜 LLM 응답(프롬프트로 어느 호출인지 가림),
같은 가짜 Workers AI · Vectorize(진입 모듈이 `env` 를 Proxy 로 감쌈). 시나리오는 이미지 · 검색(DeepInfra · Gemini · 망가진 JSON · 단일 문맥) ·
스캔(SyntheticData 의 실제 목록 · 상세 화면) · 오류(빈 본문 · 작업 없음 · 팀 없음 · rpm 소진 · 읽을 수 없는 gzip · 잘못된 URL)입니다.
`hashId()` 의 무작위 id(원본은 `ethers.Wallet.createRandom`)는 한쪽에만 나오는 주소를 나온 순서대로 자리표로 바꿔 견줍니다.

```
cd scripts/workers-diff && npm install
(cd ../../workers/apac && worker-build --release) && (cd ../../workers/proxy && worker-build --release)
node apac.mjs && node proxy.mjs          # ONLY=scan_goods_list node proxy.mjs 로 하나만
```

## 원본의 버그 (그대로 옮김 — 고칠 후보)

차분 시험이 같아야 하므로 고치지 않았습니다. 고치면 차분 시험의 기대값도 같이 바꿔야 합니다.

1. **D1 BLOB 을 pako 가 못 읽는 경우 (중요).** D1 은 BLOB 을 숫자 배열로 돌려주는데 pako 는 배열을 Uint8Array 로 바꾸지 않고 읽다가
   저장 블록(짧거나 잘 줄지 않는 글 — 예: 한국어 주소 하나인 소유자 행)을 만나면 `input.subarray is not a function` 을 던집니다.
   그 작업은 통째로 버려집니다. `ungzip(new Uint8Array(row.data))` 로 고칠 수 있습니다.
2. **이미지 결과가 저장되지 않음.** items 삽입이 `item.*` 대신 `task.digest` · `task.ref` 등을 묶어, 작업에 `digest` 가 없으면
   `D1_TYPE_ERROR: Type 'undefined' not supported` 로 끝납니다(원본 2927~2939).
3. **스캔 아이템 반복이 첫 아이템에서 멈춤.** 관계 단계가 이전 회차의 `edgeType` 으로 `env["commerce_logis_<zone>_undefined"]` 를 찾아
   던집니다(원본 5087). 그 뒤의 아이템 · 관계 저장이 일어나지 않습니다.
4. **검색 벡터 결과.** 지우기가 `results[r]` 가 아니라 함수 범위의 `results[i]`(다른 반복이 남긴 값이거나 undefined)를 써서 엉뚱한 행을 고치거나, undefined 면 던져 `rag.search.vector` 가 채워지지 않습니다(원본 6130).
5. `isDiff` 가 키 `data` 에서 `Buffer.isBuffer` 를 부르는데 `nodejs_compat` 이 없어 `Buffer` 가 없습니다(런타임에서 `typeof Buffer` = undefined 확인) — ReferenceError.
6. `cleanNumber` 의 `/./gi` 는 점이 하나라도 있으면 줄바꿈 밖의 모든 글자를 지웁니다(운송장 번호가 빈 글이 됨).
7. 이미지 갈래의 Gemini 대체 호출이 `config = null` 을 넘겨 요청 뒤 `config["response_mime_type"]` 에서 던집니다.
8. SQL 을 문자열로 이어 붙입니다(작은따옴표가 든 값은 쿼리를 깨뜨림).
9. 그 밖: `.tirm` 오타(죽은 갈래), `parseBody` 의 인자 가림과 정의되지 않은 `selectors`(쓰는 곳 없음), 다시 선언해도 지워지지 않는 `var` 에 기대는 곳들.

## 알려진 차이 (재현할 수 없거나 관찰되지 않는 것)

- gzip 압축 바이트는 pako 와 다릅니다(miniz_oxide). 푼 글은 같고, 저장 · 비교는 늘 풀어서 하므로 동작은 같습니다.
- CSS 선택자: css-select 의 확장(`:contains` · `:checked` · `:has-text` 류, `[attr!=v]`)은 SyntaxError 가 됩니다(원본은 맞춤). LLM 이 낸 선택자에만 해당.
- `new Date("…")` 는 ISO 8601 과 `YYYY/MM/DD[ hh:mm[:ss]]` 만 읽습니다(V8 의 옛 해석기가 받는 그 밖의 꼴은 NaN). 프롬프트가 ISO 꼴을 요구합니다.
- BMP 밖의 이름 있는 엔티티는 linkedom 에서 짝 없는 서로게이트 두 글 노드가 되는데, Rust 문자열은 그것을 담지 못해 Pug 줄에 U+FFFD 로 나옵니다.
- 예외 메시지 몇 곳(우리가 직접 던지는 TypeError)은 V8 의 글과 조금 다릅니다. 로그와 `console` 표에만 남습니다(시험한 경로에서는 모두 같았음).
- `hashId()`(인자 없음)는 원본이 BIP-39 니모닉으로 지갑을 만들어 그 개인 키를 쓰고, Rust 는 무작위 32바이트를 씁니다. 둘 다 무작위 id 입니다.

## 배포

`wrangler.json` 의 바인딩 · 이름 · 크론은 그대로이고, `main` 이 `build/index.js`, `build.command` 가
`cargo install -q worker-build@0.8.7 --locked && worker-build --release` 입니다. 배포 PC 에는 Rust 와 `wasm32-unknown-unknown` 표적이 있어야 합니다.
`npm run deploy`(predeploy 가 `migrations/` 를 DB 에 적용).
