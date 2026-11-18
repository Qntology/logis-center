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
  토크나이저 · 출력 투영(`TrimmedHead`) · 실행 흐름별 언어 범위(`DocLangScope`)는 PD 에 없어 앱에 남겼습니다
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
  `truncate_pug_by_tokens`)는 앱의 `utils/ai_utils.rs` · `utils/parsing.rs` 에, 저장소 · 모델을 쓰는 관계 단계 3개(`relation_join` ·
  `trade_relation_join` · `cross_lingual_title_anchors`)는 앱의 `search_relation.rs` 에 남았습니다.
- `logis-pug` 안의 코드는 경로(`crate::utils::…` · `crate::parsing` · `crate::logic`)를 그대로 쓰도록 아래 crate 의 모듈을 같은 자리에 다시 내보냅니다.
- SDS 원장은 모델 계층을 모르도록, 비전 트랙 키의 정밀도 꼬리표(SigLIP2 양자화 지문)를 앱의 `enter_scope` 가 등록합니다(`set_vision_precision`).
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

## 4단계 첫 조각: EventSink

- `commerce/src-tauri/src/sink.rs`: `EventSink`(`emit_json` · `log_progress`) · `Sink = Arc<dyn EventSink>` · `TauriSink` · `NoopSink`.
  `dyn EventSink` 에 `tauri::Emitter::emit` 과 같은 모양의 `emit` 을 두어, model · scheduler 의 `app_handle.emit(…)` · `app_handle.clone()` 호출은 그대로입니다.
- model · scheduler · utils(logger · network) 의 `tauri::AppHandle` 인자 · 필드(`LogisModel::app_handle` 포함)는 `Sink` 입니다. lib.rs 의 명령 진입점이
  `sink::tauri(&app_handle)` 로 감싸 넘기고, 시험 · CLI 는 `sink::noop()` 을 넘길 수 있습니다. `tauri::` 가 lib.rs · automation.rs 밖에 남은 곳은
  utils/network.rs 의 `#[tauri::command]` 3개뿐입니다 (Qwen 생성 루프의 `tauri::async_runtime::spawn` 9곳은 logis-llm 의 `rt::spawn` 훅으로).
- 경로 도우미(`get_kv_dir` · `get_task_log_file` …)의 쓰지 않던 `Option<&AppHandle>` 인자를 뺐습니다(37곳).

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
- Stanza · granite 순전파를 `premium-discount-nlp` 로, 스팬 NMS 를 `premium-discount-nms` 로 바꾸는 것은 하지 않았습니다. 두 저장소가 따로 갖던
  `onnxruntime(-sys)` 사본은 이 저장소 판(`commerce/crates`, ORT 1.28.0)을 기준으로 premium-discount 쪽을 다시 맞췄습니다(premium-discount
  0d83b8d: 이 판 그대로 + 기본 꺼진 `runtime-load` 기능 — 링크 대신 실행할 때 불러옴, premium-discount-nlp 의 stanza 가 켬). 남은 일은 두
  Stanza 파이프라인 · NMS 의 출력을 실제 모델로 대조한 뒤 이 앱을 그 crate 로 바꾸는 것입니다.

## 남은 단계 (3 · 4)

- **logis-search 의 나머지**: 기획서가 이 crate 에 넣은 `model/merge.rs`(병합 · 그라운딩, SigLIP2 결과를 씀)와 `model/research.rs` 의 집계는
  모델 · 저장소 호출과 섞여 있어 순수 부분을 먼저 떼어야 합니다 (`merge_rules` 로 두 규칙만 옮김).
- **commerce/app 분리**: EventSink · spawn 훅 · `CURRENT_UI_CATEGORY` 다음 걸림돌은 lib.rs 에 있는 전역 상태 2개(`ACTIVE_TASK_MEM` 6곳 ·
  `LATEST_PROGRESS_PAYLOAD` 2곳)를 model · scheduler 가 `crate::` 로 직접 읽는 것입니다. 이것들을 코어로 옮기면 store · residency · model · scheduler 를
  `commerce/app` 으로 옮기고 Tauri 셸(lib.rs 명령 · automation · TauriSink)만 남길 수 있습니다.
- **LanceDB 0.41**: premium-discount 앱이 옮긴 API 변경(테이블 add · 질의 실행)을 store.rs 에 적용합니다. 이 작업 공간은 아직 0.31 입니다.

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
  os error 1455 가 나므로, `cargo build --release … -j 3` 으로 lib → bin 을 만든 뒤 `cargo test --release --no-run --test it … -j 3` 을 돌리면 겹치지 않습니다.
- crate: 루트에서 `cargo test -p logis-schema -p logis-bias -p logis-sds -p logis-pug -p logis-parsers -p logis-search`.
