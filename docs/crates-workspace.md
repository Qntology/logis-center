# crates 작업 공간 · premium-discount 공용 엔진 통합 (2026-11-18)

기획서 "웹 분석(`analytic`) 이관 이후 `crates/` 모듈화 및 `premium-discount` 공용 엔진 통합"의 1 · 2단계, 3단계의 검색 · 생성 모델 crate,
4단계의 `EventSink` 를 반영한 기록입니다. 나머지(문서 비전 crate, `commerce/app` 분리)는 아래 "남은 단계"에 실측한 결합도와 함께 적었습니다.

## 1단계: 루트 작업 공간 · candle 포크 고정 · premium-discount 링크

| 무엇 | 어디 |
| :--- | :--- |
| 루트 Cargo 작업 공간 (`members` = `crates/logis-*` + `commerce/src-tauri`, path 의존성인 embed-cpu-kernels · ggml-kquant · onnxruntime(-sys) 는 암묵 멤버) | `Cargo.toml` |
| 자기 `[workspace]` 가 있는 boa-engine · direct-storage · cpu-embed-bench 와 `[workspace]` 없는 모바일 셸은 `exclude` (모바일을 빼지 않으면 그 폴더의 cargo 가 실패) | `Cargo.toml` |
| `[patch.crates-io]` · `[profile.*]` 는 루트로 (멤버의 것은 무시됨). `Cargo.lock` 도 루트로 | `Cargo.toml`, `Cargo.lock` |
| `target-cpu=native` 와 산출물 폴더(`commerce/src-tauri/target`)를 루트 config 한 곳에. 배치 파일 · CI 캐시 · Tauri 번들 경로가 그대로이고, config 를 여러 곳에 두면 rustflags 배열이 이어 붙어 전부 다시 빌드됨 | `.cargo/config.toml` |
| candle-rocm-vulkan 포크를 `branch = "master"`(잠금 60ec257) 대신 premium-discount · raon-zena 와 같은 커밋 `ebaf93d` 로 고정 | 루트 `[patch.crates-io]` 3줄과 `[workspace.dependencies]` 의 `candle-rocm`(logis-llm 이 씀) |
| CI 의 포크 path 전환이 루트 4줄(`candle-rocm` · 패치 3줄)을 바꿈, 캐시 키는 루트 `Cargo.lock` | `.github/scripts/use-fork-path.sh`, `env-report.sh`, `app-tests.sh`, 워크플로 |

premium-discount-* (rev `9925011`, raon-zena 와 같음):

- **hashId → `premium-discount-identity`**: ethers-core · ethers-signers 를 뺐습니다. 메시지 해시가 개인 키 범위 밖일 때의 예비
  재해시(`"logis-rehash:" + 0x 해시`, 네 번)와 `0xdead…` 값까지 예전과 같습니다. `tests/it/utils_identity.rs` 의 ethers 실행 벡터로 확인합니다.
- **SDS 통계 → `premium-discount-sds`**: `Welford` · `DecayShape` · `decay_shape` · `DecayStat` · `ConfusionStat` · `tie_verdict` · `TIE_*` 를
  재수출합니다. 필드 이름 · serde 기본값이 같아 기존 원장 파일을 그대로 읽습니다. 다른 점은 표본분산이 반올림으로 음수가 되면 0 으로 자르는 것 하나입니다.
- **이중 센터링 · 배타 배정 → `premium-discount-sds::matrix`** (`double_center_matrix` = `double_center`, `exclusive_assign_by_score_adaptive` =
  `exclusive_assign_adaptive`): 무작위 행렬 20만 개 차등 시험에서 유한한 점수는 결과 · 진단 · `allow_tie` 호출 순서까지 비트 단위로 같았습니다.
  NaN · ±inf 칸이 섞인 행렬 10.9만 개에서는 예전 코드가 2,254번(약 2 %) 정렬 비교자 패닉을 냈고(Rust 1.81+, 릴리스는 `panic = "abort"`),
  PD 판은 그런 칸을 후보 · 평균에서 빼 0번이었습니다.
- **어휘 잘라 내기 규칙 → `premium-discount-quant::trim`**: 토큰 종류 · `Langs` · `TrimRule` · `select_from` 은 토큰 단위로 같습니다.
  토크나이저 · 출력 투영(`TrimmedHead`) · 실행 흐름별 언어 범위(`DocLangScope`)는 PD 에 없어 logis-llm(`models/common/vocab_trim.rs`)에 남겼습니다
  (`Langs::of_document` → `langs_of_document`, `TrimRule::from_env` → `rule_from_env`).

## 2 · 3단계: 웹 분석 제거 · crate 6종

### 웹 분석(analytic) 제거

premium-discount 로 옮긴 트랙(`premium-discount-analytic`, www `/analytic`, `tracker.js`, 앱 `analytic.rs`)을 이 앱에서 지웠습니다.

- 백엔드: `analytic.rs` · `model/query_analytic.rs` · `structure_pending_analytics` 명령 · 합성 평가의 `eval_analytic_ingest`,
  `ai_search_complex` 의 분석 분기(페이지 · 행동 · 부정 조건 · 최근성 감쇠 · 유형 순서 · 리포트), `search_relation` 의 분석 고정점 · 방문 여집합,
  스케줄러의 `analytic_extraction` 경로(남은 옛 작업은 처리하지 않고 끝냄), 분석 프롬프트 6개, SDS 의 분석 전이 통계 함수.
- 커머스 · 무역 질의가 쓰던 도우미는 옮겼습니다: 기간 어휘 근거(`period_*`) → `time_guide`, 질의 형태소 · 원형 후보
  (`tokenize_query_with_morphology` · `morphological_variants`) → `model/query_shipping.rs`.
- 프런트엔드: `modes/analytic.ts` · `modes/oauth.ts` 와 시험, 분석 탭 · 사이트 등록 버튼 · 동기화 폴링 · 결과 말풍선, 저장된 검색 모드가
  `analytic` 이면 커머스로 되돌림. tsc 0 오류, vitest 217개 통과(이전 291개 중 지운 분석 · OAuth 시험 74개 차이).
- 기존 DB 에 남은 `mode = 'analytic'` 행은 지우지 않고 격리합니다: 저장소의 유형 구분(`store::ANALYTIC_TYPES`)과 원장 이름(`Track::Analytic`)은 남겼습니다.
- 합성 평가(관계형 v2)에서 행동 로그 질의 A1–A6(144건)과 웹 로그 넣기를 뺐습니다(커머스 240 · 무역 84 남음). 옛 계획 파일의
  `analytic` 줄은 하네스가 건너뜁니다(`skipped`). 웹 로그 생성기(`weblog.py`)와 채점기의 분석 경로는 옛 결과를 읽을 수 있게 남겼습니다.

### crate

| crate | 모듈 (예전 위치) | 의존 |
| :--- | :--- | :--- |
| `logis-schema` | `canonical` (utils/canonical.rs) · `logic` (logic.rs) · `hash` (utils/hash.rs) · `text` (ai_utils 의 숫자 · 월 · 구 나누기) · `embed_space` (models/embedding.rs 의 차원 · 순전파 모드 · 코사인 눈금 표) · `envelope` (store.rs 봉투 컬럼) · `paths` (utils/paths.rs 전부: 앱 데이터 · tmp · kv · 작업 폴더 · 로그, sync_utils 의 추출 중단 신호) | premium-discount-identity |
| `logis-bias` | `bias_schema` + `bias.json` · `full_bias.json` (utils/) · `bank` (ai_utils 의 라벨 · 선입견 뱅크) · `time_guide` · `lang_utils` · `query_lang` (stanza.rs 의 질의 언어 판정 · 따옴표 구간 · 앱 언어 코드) · `entity` (scheduler/entity.rs) | logis-schema, 기능 `full-bias` |
| `logis-sds` | SDS 원장 (utils/score_dynamics.rs) | logis-schema, premium-discount-sds |
| `logis-pug` | `utils::{parsing, ai_utils, nl_convert, json_parse, json_utils, pug_utils}` · `prompts` · `js_templates` · `merge_rules` (model/merge.rs 의 순수 규칙) · `scripts` (model/lang_llm.rs 의 문자 체계) | logis-schema, logis-bias, logis-sds, premium-discount-sds |
| `logis-parsers` | `documents` (parsers/mod.rs) · `compression` (utils/compression.rs) | — |
| `logis-search` | `fusion` (search_fusion.rs) · `relation` (search_relation.rs 의 유형 · 고정점 · 연결 · 서식 질의 근거 규칙) | logis-schema, logis-bias, logis-pug |
| `logis-llm` | `models::{qwen, qwen3, qwen3_5, qwen3vl, qwen4_exp, granite, common, vision_cache, embedding}` (models/, granite 임베딩 포함) · `lang_gguf` (model/lang_gguf.rs) · `tokenizer` · `chat_template` · `openai_types` · `position_embed` · `utils::{device_utils, resources, tensor_utils, direct_loader, img_utils, crypto, misc_utils}` (utils/) · `progress` (scheduler 의 `PROGRESS_TX` · lib.rs 의 `CURRENT_UI_CATEGORY`) · `lang_resident` (lang_llm 의 올라 있는 변형) · `rt` (spawn 훅) | candle(포크) · tokenizers · logis-schema · logis-bias · logis-sds · premium-discount-quant, 기능 cuda · rocm · vulkan · metal · flash-attn · turbo-quant (앱의 같은 기능이 켬) |
| `logis-vision-doc` | `siglip2` (models/siglip2 전부: 타워 · GGUF · 사용 시 양자화 · 정밀도 지문, 문서 분류 히트맵 · 가독성 · 크롭 계획 · NMS · 값 그라운딩 · 레이아웃 · 구 캐시) | logis-llm · logis-pug · logis-schema · logis-bias · logis-sds, 기능 cuda · rocm · vulkan · metal |

- 앱의 예전 파일은 한 줄 재수출(`pub use logis_…::*;`)로 남아 호출부가 바뀌지 않았습니다. 임베딩 모델(`LogisModel`)이나 토크나이저가 필요한
  함수 5개(`status_key_banks` · `status_pivot_bank` · `canonicalize_trade_columns_with_embedding` · `extract_doc_table_headers_async` ·
  `truncate_pug_by_tokens`)는 코어(commerce/app)의 `utils/ai_utils.rs` · `utils/parsing.rs` 에, 저장소 · 모델을 쓰는 관계 단계 3개(`relation_join` ·
  `trade_relation_join` · `cross_lingual_title_anchors`)는 코어의 `search_relation.rs` 에 남았습니다.
- `logis-pug` 안의 코드는 경로(`crate::utils::…` · `crate::parsing` · `crate::logic`)를 그대로 쓰도록 아래 crate 의 모듈을 같은 자리에 다시 내보냅니다.
- SDS 원장은 모델 계층을 모르도록, 비전 트랙 키의 정밀도 꼬리표(SigLIP2 양자화 지문)를 코어 `utils/score_dynamics.rs` 의 `enter_scope` 가 등록합니다(`set_vision_precision`).
- 각 crate 는 모델 · DB · Tauri 없이 `cargo test -p logis-schema -p logis-bias -p logis-sds -p logis-pug -p logis-parsers -p logis-search` 로 몇 초 안에 돕니다.

### logis-llm (생성 모델 엔진)

- 파일은 `git mv` 로 옮겼고(이력 유지), 앱은 예전 모듈 경로에서 재수출합니다: `models/mod.rs` 의 `pub use logis_llm::models::{…}`,
  lib.rs 의 `pub use logis_llm::{chat_template, openai_types, position_embed, tokenizer}`, `model.rs` 의 `pub use logis_llm::lang_gguf`,
  `utils/mod.rs` 의 `pub use logis_llm::utils::{…}`. 옮긴 코드 안의 `crate::utils::…` 는 crate 가 같은 이름 모듈을 두고 아래 crate(사전 · 경로 · SDS)를
  같은 자리에 다시 내보내 그대로 둡니다.
- 앱 쪽에 기대던 곳은 이렇게 끊었습니다.
  - 진행 채널(`scheduler::PROGRESS_TX`)과 화면 분류(`CURRENT_UI_CATEGORY`): 정의를 `logis_llm::progress` 로 내리고 앱은 재수출 (정의는 하나).
  - RAM 압박 기준 · 스냅숏 · 힙 반환(model.rs 의 6개)과 올라 있는 언어 엔진 변형(lang_llm 의 `RESIDENT`): 순수 함수 · 상태라 아래로 내림.
  - `tauri::async_runtime::spawn` 9곳(KV 슬롯 반환 · 굽기 · 읽기 · 지연 해제): `logis_llm::rt::spawn`. 앱 `run()` 이 `set_spawner` 로
    `tauri::async_runtime::spawn` 을 등록해 예전과 같은 런타임에서 돌고, 등록이 없으면(통합 시험 · 헤드리스) crate 가 처음 쓸 때 만드는 전역
    멀티스레드 런타임에 띄웁니다 — Tauri 가 `async_runtime::set` 없이 스스로 만드는 기본 런타임과 같은 성질(부른 스레드의 런타임과 무관).
  - `lang_gguf` 는 Qwen3.5 변환 · 적재뿐이라 통째로 내리고 `pub(crate)` 18개를 `pub` 으로 (SigLIP2 양자화가 GGUF 쓰기 도우미를 씀).
  - 앱 경로 도우미(`utils/paths.rs`)는 std 만 써서 `logis_schema::paths` 로 합쳤습니다(`get_kv_dir` 등 그대로).
- GPU 기능: 옮긴 코드의 `#[cfg(feature = "rocm")]` 등은 이제 logis-llm 의 기능을 봅니다. 앱의 `cuda` · `rocm` · `vulkan` · `metal` ·
  `flash-attn` · `turbo-quant` 가 `logis-llm/…` 도 켭니다 — 빠뜨리면 컴파일 오류 없이 그 경로가 조용히 빠지므로 기능 목록은 둘이 같아야 합니다.
- candle-rocm(포크 커밋 고정)은 logis-llm 만 써서 루트 `[workspace.dependencies]` 로 옮겼습니다. 포크 path 전환(`use-fork-path.sh`)은 이제 루트
  한 파일의 4줄을 바꿉니다.
- granite 임베딩(`models/embedding`)도 candle 모델이고 앱에 기대는 곳이 경로 · GPU 정보뿐이라 logis-llm 의 `models::embedding` 으로 옮겼습니다.
  기획서는 순전파 코어를 premium-discount-nlp(granite)로 바꾸라고 하지만, 캐시 공간 태그 · 눈금 보정 · 배치 경로의 출력을 실제 모델로 대조하기
  전에는 바꾸지 않습니다(Stanza · NMS 와 같은 이유).

### logis-vision-doc (문서 비전)

- `models/siglip2` 전체를 `crates/logis-vision-doc/src/siglip2` 로 옮기고 경로를 crate 이름으로 바꿨습니다(`crate::logic` → `logis_schema::logic`,
  `crate::utils::ai_utils` · `parsing` → `logis_pug`, `score_dynamics` → `logis_sds`, `model::lang_gguf` → `logis_llm::lang_gguf`, 배치 정책 ·
  Vulkan 실행 모드 → `logis_llm::models::embedding`). 앱은 `crate::models::siglip2` 로 재수출하고, SDS 의 비전 정밀도 꼬리표 등록도 그 경로 그대로입니다.
- 기획서는 타워(vision · text · tokenizer · quant · preprocessor)를 premium-discount-vision · -sensor 로 바꾸고 서식 전용 모듈만 남기라고 하지만,
  이 앱의 ROCm q8 GGUF · 정밀도 지문 경로와 HF 로더 차이(열 순서 · antialias · bilinear · 토크나이저 패딩)를 실제 모델로 대조하기 전에는 바꾸지 않고
  타워까지 그대로 옮겼습니다.
- `quant.rs` 의 GPU 기본 정밀도(`cfg!(feature = "rocm") && !cfg!(feature = "cuda")` 이면 q8)가 이 crate 의 기능을 보므로 앱의 rocm · cuda 가
  `logis-vision-doc/…` 도 켭니다.

## 4단계: EventSink · commerce/app (헤드리스 코어) 분리

### EventSink

- `EventSink`(`emit_json` · `log_progress`) · `Sink = Arc<dyn EventSink>` · `NoopSink` 는 코어 `commerce/app/src/sink.rs`, Tauri 창으로 보내는
  `TauriSink` · `sink::tauri(&app_handle)` 는 셸 `commerce/src-tauri/src/sink.rs`(코어 `sink` 를 같은 이름으로 재수출)에 있습니다.
  `dyn EventSink` 에 `tauri::Emitter::emit` 과 같은 모양의 `emit` 을 두어, model · scheduler 의 `app_handle.emit(…)` · `app_handle.clone()` 호출은 그대로입니다.
- model · scheduler · utils(logger · network) 의 `tauri::AppHandle` 인자 · 필드(`LogisModel::app_handle` 포함)는 `Sink` 입니다. lib.rs 의 명령 진입점이
  `sink::tauri(&app_handle)` 로 감싸 넘기고, 시험 · CLI 는 `sink::noop()` 을 넘길 수 있습니다.
- 경로 도우미(`get_kv_dir` · `get_task_log_file` …)의 쓰지 않던 `Option<&AppHandle>` 인자를 뺐습니다(37곳).

### commerce/app (crate `logis-app`)

- `commerce/src-tauri/src` 의 46개 파일(model · store · scheduler · stanza · search_fusion · search_relation · logic · prompts · js_templates · parsers ·
  models · utils · sink)을 `git mv` 로 `commerce/app/src` 에 옮겼습니다. 셸에는 lib.rs(명령 49개 · `run()`), automation.rs(브라우저
  자동화), main.rs, sink.rs(`TauriSink`)만 남았고, lib.rs 가 코어 모듈을 예전 이름으로 재수출해(`pub use logis_app::{model, store, …}`) 명령의
  경로가 그대로입니다(통합 시험은 5단계에서 commerce/app 으로 옮김). 코어의 의존성 그래프에는 tauri 가 없습니다(`cargo tree -p logis-app -i tauri` → 없음).
- 코어가 셸에 기대던 곳: lib.rs 전역 `ACTIVE_TASK_MEM` · `LATEST_PROGRESS_PAYLOAD`(정의를 코어 루트로, 셸은 재수출), automation.rs 의
  `LAST_DETECTED_STATE`(코어 `browser_state`, URL 도우미가 읽음), utils/network.rs 의 `#[tauri::command]`(코어는 평범한 함수, 등록된 두 명령
  `get_local_network_prefix` · `get_my_full_ip` 는 셸 lib.rs 의 같은 이름 · 인자 껍데기 — 등록되지 않던 `get_deterministic_cert` 는 함수만).
- 코어 코드는 `#[cfg(feature = …)]` 를 보지 않습니다. 셸의 GPU · 가속 기능과 `full-bias` 는 `logis-app/…` 을 켜고, 코어가 다시 logis-llm ·
  logis-vision-doc · logis-bias 의 같은 기능을 켭니다(헤드리스로 `-p logis-app --features rocm` 도 같은 경로).
- 셸 매니페스트에서는 코어만 쓰는 의존성(lancedb · arrow · boa_engine · onnxruntime · whatlang · rcgen · ndarray · num · encoding_rs · windows-sys ·
  logis-* · premium-discount-*)을 뺐습니다. 셸 코드 · 시험이 쓰지 않던 예전 의존성(webrtc 등)은 5단계에서 뺐습니다.
- 자연어 검색(`ai_search_complex` 1,538줄)과 그 도우미 29개(범위 필터 · dexie 계획/평가 · 저장 축 맞추기 · 조건 그라운딩 · 무역 질의 탐침 ·
  질의 융합, 정적 캐시 2개)는 코어 `query` 모듈로 옮겼습니다(셸 lib.rs 5,557 → 2,517줄). 본문의 `tauri::AppHandle` 은 진행 이벤트를 보내는 데만
  쓰였으므로 같은 이름의 `Sink` 매개변수가 되었고(`use tauri::Emitter` 1곳 삭제, `sink::tauri(&app_handle)` 4곳 → `app_handle.clone()`),
  `State<AppState>` 는 `&AppState` 입니다. 셸의 `#[tauri::command] ai_search_complex` 는 같은 이름 · 인자로 코어를 부르는 껍데기이고,
  `AppState` · `IS_SEARCHING` 은 코어 루트에 있습니다(셸 재수출, 통합 시험 경로 `tauri_app_lib::AppState` · `ground_trade_conditions` 그대로).
- 나머지 명령 본문(`upsert_items` · `reindex_pending_embeddings` · `download_model` 등)은 5단계에서 코어 `commands` 로 옮겼습니다.

## 5단계: LanceDB 0.41 (키 직접 정렬) · 명령 본문 · 통합 시험 commerce/app · 셸 의존성

### LanceDB 0.41 — 키를 직접 정렬해 예전 출력 그대로

- 결정: 삽입 순서를 받아들여 다시 색인하는 대신, 맵 순서가 바깥으로 드러나는 곳에서 키를 정렬해 0.31 때(serde_json 맵 = BTreeMap)와 같은
  출력을 지킵니다. 코어는 `lancedb = "0.41"`(lance 14.0.2 · datafusion 54.1.0)입니다. preserve_order 에서 `Map` 은 IndexMap(넣은 순서 · 문서
  순서)이고 `Map::remove` 가 swap_remove 라 지운 뒤의 순서도 바뀝니다.
- 도우미 `crates/logis-schema/src/json_order.rs`(코어는 `logis_app::json_order` 로 재수출): 순회 `sorted_iter` · `sorted_keys` · `sorted_values` ·
  `sorted_iter_mut` · `into_sorted`, 직렬화 `Sorted(&Value)` · `SortedMap` · `SortedSlice`(Serialize · Display) · `to_string_sorted`, 값 정렬
  `sorted` · `sort_all`. 순서는 BTreeMap 과 같고(키 바이트 순), preserve_order 가 꺼진 빌드에서도 결과가 같습니다(단위 시험 2개).
- 고친 곳(48개 파일):
  - 순회 156곳: 맵을 돌며 문장 · 청크 · 프롬프트 · 간선 · 스키마 목록을 만드는 곳(`nl_convert`, 프롬프트 필드, 스키마 카테고리 · 필드, 자리 표시자
    복원 경로 …).
  - 직렬화 63곳: 맵이 든 Value 를 `to_string` · `{}` · `serde_json::to_string` 으로 쓰는 곳(색인 문장 · 텍스트 캐시 키 · 프롬프트 속 JSON ·
    저장소 data 열 · 원장 · 로그). 타입이 정해진 구조체 · `Vec<String>` · HashMap 은 serde_json 맵이 아니라 그대로입니다.
  - 원천: 파싱한 값(`from_str::<Value>` → `.map(json_order::sorted)`), 서식 사전 `BIAS_DICT`(파싱 · 오버레이 직후 `sort_all`), gzip 저장
    (`logis-parsers` `compression.rs`, `sort_all_objects`).
  - 화면 경계: 셸 명령 껍데기가 Value 인자는 들어올 때, Value 결과는 나갈 때 정렬하고 `TauriSink::emit_json` · 셸의 직접 `emit` 도 정렬합니다
    (웹뷰 · HTTP 본문의 키 순서도 예전과 같음).
- 빠뜨리지 않게 찾은 방법:
  - Display 감사: serde_json 사본에서 `impl Display for Value` 를 끄고(`#[cfg(any())]`) 별도 작업 트리 · 산출물 폴더에서
    `cargo check --workspace --lib --bins` 를 돌리면 Value 를 `{}` · `.to_string()` 으로 쓰는 곳이 모두 컴파일 오류로 드러납니다. 코어 25곳과 명령을
    옮긴 뒤 4곳을 고쳤고, 숫자 · 한 키짜리처럼 순서가 없는 곳은 두었습니다.
  - 순회: 스캐너로 `.iter()` · `.keys()` · `.values()` · `for (k, v) in` 후보를 찾아 바꾼 뒤, 맵이 아니었던 것(Vec · HashMap · 슬라이스)은 컴파일
    오류로 가려 되돌렸습니다.
  - 골든 덤프(임시 시험, 커밋 안 함): 키 순서에 흔들릴 수 있는 순수 함수 약 45종(자연어 변환 · 청크 · 자리 표시자 · 병합 · LLM 응답 파싱 · 관계
    간선 · 프롬프트 9종 · 스키마 삼중 · 검색 계획)을 입력마다 키 삽입 순서 3변형(정렬 · 역순 · 섞음)으로 돌려 7,303줄을 씁니다. 기준은 바꾸기 전
    0.31 빌드의 덤프입니다.
- 결과: 0.31 + 정렬은 기준과 0줄 차이이고, 0.41 + 정렬은 세 변형이 모두 같으며 기준과도 같습니다(다른 12줄은 하네스가 검색 계획을 정렬 없이
  찍은 `plan_order_*` 줄로, 앱에서는 명령 껍데기가 나갈 때 정렬). 색인 문장(임베딩 입력) · 텍스트 캐시 키 · 프롬프트가 예전과 같으므로 다시 색인하지
  않고 기존 벡터 · 해시를 그대로 씁니다. 0.41 에서 실패하던 순서 시험 4개를 포함해 `it` 286 통과.
- 앞으로: preserve_order 가 켜져 있으므로 새 코드가 `serde_json::Map` 을 돌거나 Value 를 문자열로 만들 때는 `json_order` 를 씁니다(그냥 쓰면
  넣은 순서가 나옴).

### 명령 본문 → `logis_app::commands`

- 셸 lib.rs 에 남아 있던 명령 본문 32개(`upsert_items` · `reindex_pending_embeddings` · `download_model` · 문서 조회 · 삭제 · `proxy_fetch` …)와
  `ActiveTaskQuery` 를 `commerce/app/src/commands.rs` 로 옮겼습니다. 본문은 그대로이고 시그니처만 바뀝니다: `State<AppState>` → `&AppState`,
  `tauri::AppHandle` → 같은 이름의 `Sink`, `sink::tauri(&app_handle)` → `app_handle.clone()`.
- 셸 lib.rs(2,517 → 901줄)에는 같은 이름 · 인자의 `#[tauri::command]` 껍데기(`&state` 와 `sink::tauri(&app_handle)` 을 넘김), 창(크기 · 위치 ·
  끌기 · 커서 통과) · 브라우저 실행 · 상태 명령, `run()` 만 남았습니다. 통합 시험용 헤드리스 Tauri 앱(`build_headless_app` · `eval_ai_search`)은 지웠습니다.
- `logis_app::init_headless()`: 앱 `run()` 이 하는 엔진 초기화(언어 엔진 로더 등록 · KV 굽기 작업자)를 Tauri 없이 할 때(시험 · 합성 평가 · CLI) 부릅니다.

### 통합 시험 · 합성 평가 → commerce/app (기획서 5단계)

- `commerce/src-tauri/tests/{it,fixtures}` → `commerce/app/tests/`(git mv), `tauri_app_lib::` → `logis_app::`(314곳). 시험 exe 는 logis-app 만
  링크합니다(tauri 없음). 그래서 시험 exe 에 comctl32 v6 매니페스트를 넣던 셸 build.rs 의 `rustc-link-arg-tests` 도 뺐습니다(시험 대상이 없는
  패키지가 그 지시를 내면 cargo 가 빌드를 멈춤).
- 합성 평가(`synthetic_eval.rs`): 헤드리스 Tauri 앱과 `listen` 두 개 대신 시험용 `EventSink`(`EvalSink`: `translit-cache-query` 에 빈 응답,
  `LOGIS_SYNTH_STOP_ON` 문구가 든 `task-console-log` 에서 취소)와 `init_headless()`, `logis_app::query::ai_search_complex(&state, sink, …)` 를 직접 부릅니다.
- 실행: `cargo test -p logis-center -p logis-app --test it`(작업 폴더 무관). 시험 대상은 logis-app 에만 있고, `-p logis-center` 는 데스크톱 앱과
  같은 기능 집합으로 빌드해 셸 빌드와 의존성 산출물을 같이 쓰게 할 뿐입니다. CI(`cross-verify.yml` · `app-tests.sh` · 픽스처 경로)와
  `commerce/test_cargo*.bat` · `_verify_rocm.bat` · `test_linux.sh` 를 이렇게 바꿨습니다. 시험 exe 자리(`target/debug/deps/it-*.exe`)는 그대로라
  `SyntheticData/eval/run_eval.bat` 은 고칠 것이 없습니다.
- 시험이 쓰는 crate(candle-nn · candle-transformers · dirs · ggml-kquant · memmap2 · mimalloc · nvml-wrapper · safetensors · sysinfo, 리눅스
  io-uring)는 logis-app 의 `[dev-dependencies]` 입니다.

### 셸 의존성 정리

- 셸 코드(lib.rs · automation.rs · sink.rs · main.rs)가 직접 쓰는 것만 남겼습니다: tauri 와 플러그인 3개(dialog · fs · shell), logis-app ·
  logis-llm, serde · serde_json · tokio · anyhow · regex · futures · uuid · chrono · reqwest · mimalloc · once_cell, 브라우저 자동화 fantoccini ·
  chromiumoxide · which.
- 뺀 것: webrtc · drag · ahash · minijinja · ego-tree · libc · any_ascii · embed-cpu-kernels · base64 · url, candle-core · candle-nn ·
  candle-transformers · candle-flash-attn(GPU 기능은 logis-app → logis-llm 이 켜므로 셸 기능은 이제 `logis-app/…` 만), 시험 전용 image ·
  tokenizers · scraper · sysinfo · dirs · safetensors · nvml-wrapper · half · memmap2 · rand · rand_chacha · ggml-kquant, 플랫폼별 windows ·
  direct-storage(Windows) · io-uring(리눅스) · metal(맥), 그리고 한 번도 등록하지 않은 템플릿 잔재 tauri-plugin-opener(capabilities 의
  `opener:default` 도 뺌. 프런트엔드 package.json 의 `@tauri-apps/plugin-opener` 는 import 하는 곳이 없지만 npm 쪽이라 두었음). ethers-* 는
  1단계에서 이미 뺐습니다. Cargo.lock 의 패키지는 1,202 → 1,109 개입니다.
- 남는 crate 의 동작이 바뀌지 않는지: `cargo tree -e normal,build -f "{p}|{f}"` 를 Windows(CPU · ROCm · 기본 기능) · 리눅스(Vulkan · ROCm) ·
  맥(Metal) 대상으로 앞뒤 비교했습니다. 기능이 줄어든 것은 webrtc 가 켜던 암호 crate(der · ecdsa · elliptic-curve · sec1 · signature · hmac ·
  subtle · zeroize · crypto-common …의 `std` · `alloc` · `pem`), Windows · 맥 바인딩 기능, syn 1 처럼 컴파일 시점 API 뿐이고, image · tokenizers ·
  scraper · minijinja · rand · half · candle 등 동작에 닿는 crate 의 기능은 그대로입니다. 암호 crate 를 쓰는 premium-discount-identity(k256,
  `hash_id`)는 통합 시험의 ethers 벡터 시험(`utils_identity`)이 확인합니다.

## 기획서와 다른 점 (코드 · 저장소로 확인)

- 기획서의 candle 커밋 `ebaf93db1c65fe2fb3bf3ebde3fc2bdab07ff311` 은 포크에 없습니다. 실제 커밋은 `ebaf93db065f22be67a9a4a250eec5a212e143fe` 이고,
  포크에는 candle-transformers · candle-kernels · candle-ug 가 없어 그 패치 줄은 넣지 않았습니다(넣으면 해석이 실패). candle-transformers 는
  crates.io 0.11.0 이고 그 의존성 candle-core · candle-nn 이 패치로 포크에 합쳐집니다.
- "`score_dynamics.rs` 를 `pub use premium_discount_sds::*` 로 교체, 원장 형식 100 % 호환"은 사실이 아닙니다. logis 는 전역 스코프 API
  (`enter_scope` · `record_*` · `flush` · `load`, recipe `sds-v3`, 원장 4개)이고 PD 는 `Ledger` 구조체(recipe `zena-sds-v1`)이며,
  `ScopeStat` 의 시각 칸(`updated_at` ↔ `updated_ms`)과 logis 고유 레코드(카테고리 · 공간 · 검색 필드)가 다릅니다. 같은 것만 재수출했고,
  원장 계층은 `logis-sds` 라는 crate 하나를 더 만들어 옮겼습니다(텍스트 파이프라인이 그 위에 있어야 해서).
- `scheduler/entity.rs` 는 서식 판정(`bias_schema`)을 써서 `logis-schema` 가 아니라 `logis-bias` 에 두었습니다.
- `utils/img_utils.rs` 는 이미지를 candle 텐서로 바꾸는 VLM 전처리라 "모델 의존 없음" 인 `logis-parsers` 가 아니라 `logis-llm` 으로 옮겼습니다.
- 기획서의 logis-llm 목록(models 7종 · position_embed · chat_template · openai_types · tokenizer · tensor_utils · device_utils · direct_loader)만으로는
  컴파일되지 않습니다. 모델이 KV 배치(`utils::resources` 82곳) · KV 블록 암호화(`crypto`) · 어휘 판정 도우미(`misc_utils`) · Qwen3.5 GGUF
  변환(`model/lang_gguf.rs`)을 쓰므로 함께 옮겼습니다. `vocab_trim.rs` 는 이미 premium-discount-quant 규칙 위의 얇은 층이라 그대로 옮겼습니다.
- Analytic 의 이관처는 `premium-discount-nav` · `-nms` 가 아니라 `premium-discount-analytic` 입니다(앞 작업에서 만듦).
- "lancedb 0.41 로 통일"은 저장소 코드만의 일이 아닙니다. lancedb 0.41 의 매니페스트가 `serde_json = { features = ["preserve_order"] }` 를
  요구하고(0.31 은 없음), Cargo 기능은 합쳐지므로 앱 전체의 `serde_json::Map` 이 정렬 순서(BTreeMap)에서 삽입 순서(IndexMap)로 바뀝니다.
  0.41 로 올려 보니 store.rs 는 고치지 않고 컴파일됐지만(디버그 9분) `it` 에서 키 순서에 기대는 시험 4개가 실패했습니다:
  JSON → 자연어 문장 순서(`utils_schema::json_to_natural_language_and_chunk_split`), 스키마 카테고리 · 필드 순서(`schema_field_counts_per_page_type` ·
  `trade_schema_triples_load_base_plus_overlay`), 자리 표시자 복원 경로 순서(`utils_text::placeholder_drop_and_informative_restore`).
  시험만의 문제가 아니라 색인 문장(임베딩 입력 · 텍스트 캐시 키)과 프롬프트의 필드 순서가 바뀌는 동작 변경이라 처음에는 0.31 로 되돌렸고,
  5단계에서 순서가 드러나는 곳의 키를 직접 정렬해 0.31 과 같은 출력을 지키며 0.41 로 올렸습니다.
- Stanza · granite 순전파를 `premium-discount-nlp` 로, 스팬 NMS 를 `premium-discount-nms` 로 바꾸는 것은 하지 않았습니다. 두 저장소가 따로 갖던
  `onnxruntime(-sys)` 사본은 이 저장소 판(`commerce/crates`, ORT 1.28.0)을 기준으로 premium-discount 쪽을 다시 맞췄습니다(premium-discount
  0d83b8d: 이 판 그대로 + 기본 꺼진 `runtime-load` 기능 — 링크 대신 실행할 때 불러옴, premium-discount-nlp 의 stanza 가 켬). 남은 일은 두
  Stanza 파이프라인 · NMS 의 출력을 실제 모델로 대조한 뒤 이 앱을 그 crate 로 바꾸는 것입니다.

## 남은 단계

- **logis-search 의 나머지**: 기획서가 이 crate 에 넣은 `model/merge.rs`(병합 · 그라운딩, SigLIP2 결과를 씀)와 `model/research.rs` 의 집계는
  모델 · 저장소 호출과 섞여 있어 순수 부분을 먼저 떼어야 합니다 (`merge_rules` 로 두 규칙만 옮김).
- **SigLIP2 타워 · 임베딩 · Stanza · NMS 를 premium-discount 로**: 실제 모델 출력 대조 뒤.
- **LanceDB 0.41 의 기존 테이블**: 0.31(lance 8)이 쓴 테이블을 0.41(lance 14)이 그대로 읽고 이어 쓰는 것은 premium-discount 앱에서 확인했습니다.
  이 앱의 실제 데이터 폴더로는 아직 열어 보지 않았습니다(시험은 새 임시 폴더를 씀).

## 검증 (2026-11-18, 이 PC: Windows · RAM 16 GB)

- 1단계만: release ROCm `cargo build`(lib + bin) 통과 (51분 56초 — ethers 를 빼며 호스트 syn 기능이 바뀌어 거의 전체 재빌드).
- 1~4단계 합친 상태:
  - debug `cargo check --no-default-features --features rocm --lib --bins --test it` 통과.
  - release ROCm `cargo build`(lib + bin) 통과 (28분 50초).
  - CI(windows-cpu)와 같은 조건(debug · CPU · `CARGO_PROFILE_DEV_DEBUG=0`): `cargo test --test it` 286 통과 · 0 실패 · 65 무시,
    lib 단위 시험(`vocab_trim stable_rank store::`) 8 통과. 시험 수는 388 → 375 로, 지운 분석 시험 13개(analytic_page 10 · search_relation 3)만큼.
  - crate 단위 시험(`-p logis-schema … -p logis-search`) 통과, 프런트엔드 tsc 0 오류 · vitest 217 통과(지운 분석 · OAuth 시험 74개 제외).
- logis-llm 분리 뒤 (CI 와 같은 조건): `cargo test --test it` 286 통과 · 0 실패 · 65 무시(옮기기 전과 같음), lib 단위 시험(앱 + logis-llm
  함께) 8 통과, `cargo test -p logis-llm` 단독 6 통과 · 1 무시, 도메인 crate 시험 통과. 단독 빌드에서 windows-sys `Win32_Foundation` 이 빠져
  있던 것이 드러나 매니페스트에 넣었습니다(앱 빌드에서는 다른 crate 가 켜 주어 보이지 않음).
- 임베딩 · logis-vision-doc 분리 뒤 (같은 조건): `it` 286 통과 · 0 실패 · 65 무시, lib 단위 8 통과, `cargo test -p logis-llm -p logis-vision-doc`
  단독 21 통과 · 3 무시(logis-vision-doc 의 ROCm 전용 시험 1개는 CPU 빌드에서 빠짐).
- logis-llm · logis-vision-doc 의 GPU 기능 경로: 별도 산출물 폴더에서 `cargo check -p logis-llm -p logis-vision-doc --lib --tests` 를
  ROCm + flash-attn + turbo-quant, Vulkan + flash-attn + turbo-quant 로 각각 통과(기존 경고만).
- commerce/app 분리 뒤 (같은 조건): `it` 286 통과 · 0 실패 · 65 무시, lib 단위 8 통과(코어 3 · logis-llm 5, 셸 lib 에는 단위 시험이 없음).
  통합 시험 하나가 `include_str!("../../src/model/lifecycle.rs")` 로 소스를 읽고 있어 코어 경로로 고쳤습니다.
  코어만 따로(`cargo check -p logis-app --lib --tests`, 셸 · tauri 없이) 통과 — 그때 새 산출물 폴더라 onnxruntime-sys 가 GitHub 404 뒤 NuGet
  예비 경로로 ORT 1.28.0 을 받는 것도 다시 확인했습니다.
- 세 단계(logis-llm · logis-vision-doc · commerce/app)를 합친 상태의 release ROCm `cargo build`(lib + bin) 통과 (27분 12초).
- 검색 본문을 코어 `query` 로 옮긴 뒤 (CI 와 같은 조건): `it` 286 통과 · 0 실패 · 65 무시, lib 단위 8 통과, crate 시험 통과. 첫 빌드에 그대로
  컴파일됐고(코어 · 셸 모두 `#[cfg(feature)]` 가 없어 CPU 빌드가 모든 경로를 봄), 옮기며 남은 쓰지 않는 import 4개만 지웠습니다.
- 5단계 (CI 와 같은 조건):
  - 키 정렬을 넣은 0.31: `it` 286 통과 · 0 실패 · 65 무시, 골든 덤프가 기준과 0줄 차이.
  - lancedb 0.41: `it` 286 통과 · 0 실패 · 65 무시(0.41 에서 실패하던 순서 시험 4개 포함). 골든은 세 변형이 같고 기준과는 `plan_order_*` 12줄만
    다름(위). Display 감사(`cargo check --workspace --lib --bins`, Display 를 끈 serde_json 사본) 통과.
  - 명령 · 시험을 옮긴 뒤: `cargo test -p logis-center -p logis-app --test it` 286 통과 · 0 실패 · 65 무시(시험 exe 에 tauri 없음), lib 단위 8 통과.
  - 셸 의존성 정리 뒤: `it` 286 통과 · 0 실패 · 65 무시, lib 단위 8 통과(코어 3 · logis-llm 5), logis-llm · logis-vision-doc 단독 21 통과 · 3 무시,
    crate 시험 통과, 골든 덤프가 0.41 때와 바이트 단위로 같음. 암호 crate 기능이 바뀌어 digest 를 쓰는 lance · datafusion 까지 다시 빌드됐습니다
    (debug 7분 30초).
- 함정: 같은 `target/debug` 에 `cargo check`(ROCm, 메타데이터만)와 CPU 빌드를 섞으면 boa 의 타입이 엉뚱하게 읽혀 `str` 크기 오류 · "rlib 형식이
  없음" 오류가 났습니다(폴더를 비우고 다시 빌드하면 사라짐). debug 정보를 켠 채 `it` 시험 exe 를 링크하면 PDB 가 1.7 GB 를 넘어 LNK1318(LIMIT 12) —
  CI 처럼 `CARGO_PROFILE_DEV_DEBUG=0` 을 씁니다.
- 2026-11-18 08:00 쯤부터 `github.com/microsoft/onnxruntime` 저장소가 404 라(07:20 엔 받아졌음) onnxruntime-sys 빌드 스크립트의 내려받기가
  실패했습니다(첫 CI 실행 44203533916 의 빌드 단계 실패가 모두 이것). 그래서 `commerce/crates/onnxruntime-sys/build.rs` 에 예비 경로를 더했습니다:
  GitHub 배포판을 받지 못하면 같은 판을 NuGet `Microsoft.ML.OnnxRuntime` 에서 꺼내 배포판과 같은 `include/` · `lib/` 로 둡니다(CPU 판만,
  win-x64 헤더 · DLL · LIB 가 배포판과 바이트 단위로 같음, 리눅스 SONAME `libonnxruntime.so.1` · 맥 `libonnxruntime.1.dylib` 이름도 둠).
  지금 상태에서 새 빌드 폴더로 빌드 → 링크 → 실행해 `ONNX Runtime 1.28.0` 을 확인했습니다.
- 덧붙여: Windows 11 은 `System32\onnxruntime.dll`(1.17.1)을 갖고 있고, 암시적으로 링크한 DLL 은 PATH 보다 System32 를 먼저 찾으므로 이 앱 판(빌드 때
  링크)은 DLL 이 실행 파일 옆에 없으면 그 옛 판을 불러옵니다(시험 exe 에서 1.17.1 로 확인). premium-discount 의 `runtime-load` 가 전체 경로로만
  여는 이유가 이것입니다.

## 빌드 · 시험

- 앱: 예전처럼 `commerce/*.bat` (작업 폴더 `commerce/src-tauri`). 이 PC(RAM 16 GB)는 마지막 LTO 링크 두 개(lib · bin, 시험 exe)가 겹치면
  os error 1455 가 나므로, `cargo build --release … -j 3` 으로 lib → bin 을 만든 뒤
  `cargo test --release --no-run -p logis-center -p logis-app --test it … -j 3` 을 돌리면 겹치지 않습니다.
- 통합 시험: 어디서든 `cargo test -p logis-center -p logis-app --test it --no-default-features [--features rocm|vulkan|metal]`
  (`commerce/test_cargo*.bat` 가 이것). Windows debug 는 `CARGO_PROFILE_DEV_DEBUG=0`(PDB 한도, 위 함정).
- crate: 루트에서 `cargo test -p logis-schema -p logis-bias -p logis-sds -p logis-pug -p logis-parsers -p logis-search`.
