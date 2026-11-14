# 합성 데이터 평가

합성 데이터를 실제 파이프라인에 태워 정답과 비교하고, 앱이 AppData 에 남긴 SDS 실행 파일과 교차 확인합니다. 브라우저 · 프론트엔드 · 창 없이 앱과 같은 `process_task` / `ai_search_complex` 를 부릅니다 (`commerce/src-tauri/tests/it/synthetic_eval.rs`).

## 파일

| 파일 | 역할 |
|---|---|
| `make_plan.py` | 평가 계획(JSONL). 커머스 페이지 · 무역 문서 · 페이지 이미지 · 검색 질의 · 정답 레코드 색인 |
| `run_eval.bat` | ROCm 테스트 바이너리 실행. 실험마다 `%LOCALAPPDATA%\logis-center-eval\<이름>` 을 씁니다 |
| `rasterize_trading.py` | 무역 PDF → 페이지 PNG + 단어 위치. 앱은 PDF 를 이미지로 바꾸지 않으므로 SigLIP2 경로를 재려면 필요합니다. 렌더러는 pypdfium2(BSD-3-Clause / Apache-2.0) — 예전 PyMuPDF(AGPL-3.0)는 쓰지 않습니다 |
| `make_vision_cases.py` | 비전 게이트 합성 평가 세트: 페이지에 열화(흐림 · 도장 · 잉크 얼룩 · 서명 획)를 입힌 출처 판독성 사례, 인명 잔해('-Jane'), 11개 언어 팩(`trading/ml/packs`)의 인쇄 라벨 → 축 |
| `score_vision_gates.py` | 위 세트의 채점: 판독성 게이트 재현율 · 오폐기율, 라벨 라우팅 정답률, 인명 잔해 |
| `score_pages.py` | 페이지 유형 · 목록/상세 판정 (실행 로그에서 읽음) |
| `score_commerce.py` · `score_trading.py` | 저장된 아이템과 정답 비교. `score_commerce.py --pool` 은 `{"id":"dump","kind":"dump"}` 계획으로 남긴 저장소 끝 상태로 찾습니다. 상세가 앞서 저장된 목록 문서에 합쳐지면 그 페이지의 결과에는 아이템이 남지 않기 때문입니다 |
| `score_vision.py` | SigLIP2 서식 판정 · 크롭 커버리지 |
| `score_vision_quant.py` | SigLIP2 정밀도 계획끼리 비교: 원본 F32 기준 히트맵 AP(쪽 부트스트랩 구간) · 서식 판정 · 크롭 · 문서번호 회수 · 시간 |
| `score_verdict.py` | 이미지 경로의 최종 서식(언어 모델 재판정 포함). `LOGIS_SYNTH_STOP_ON="Document identified as"` 로 돌린 로그를 읽습니다 |
| `score_search.py` | 질의 종류별 MRR · Recall@10 · nDCG@10. `--seeds` 를 주면 프론트엔드 조건 필터를 거친 화면 순위로 잽니다 |
| `apply_plans.py` | `main.ts` 의 `executeDexiePlan` 을 옮긴 것. 잘못된 조건이 정답을 지우는 실패는 후보 순위에 보이지 않습니다 |
| `fusion_lab.py` | 응답의 트랙별 순위로 융합 가중치를 재실행 없이 비교합니다 |
| `sds_crosscheck.py` | SDS 축이 실제로 맞고 틀림을 가르는지(AUC), 원장과 실행 파일이 같은 사실을 말하는지 |

## 돌리는 법

ROCm 빌드만 씁니다. 빌드는 `commerce\test_cargo_rocm.bat --no-run` 입니다.

```bat
python make_plan.py commerce --split dev --out plan.jsonl
run_eval.bat 실험이름 plan plan.jsonl result.jsonl run.log
python score_commerce.py result.jsonl --split dev
```

검색은 정답 레코드를 앱의 색인 경로로 바로 넣고 재면 추출과 분리됩니다.

```bat
python make_plan.py seeds --split dev --out seeds.jsonl
python make_plan.py search --split dev --out search.jsonl
run_eval.bat 실험이름 plan seeds.jsonl seeded.jsonl seed.log
run_eval.bat 실험이름 seedids seeds.jsonl seed_map.json ids.log
run_eval.bat 실험이름 plan search.jsonl search_out.jsonl search.log
python score_search.py search_out.jsonl --seed-map seed_map.json --seeds seeds.jsonl --dump detail.json
python sds_crosscheck.py search <평가 폴더> search_out.jsonl detail.json
```

임베딩 판정만 빠르게 재려면 `LOGIS_SYNTH_STOP_ON="Determined Detail Page"` 를 주고 돌립니다. 언어 모델 앞에서 멈춥니다.

### 12개 언어 시나리오 (Stanza)

`queries_12lang.yaml` 은 같은 정답 집합의 질의를 12개 언어(en · ko · de · es · fr · it · pt · nl · cs · ar · ja · zh-hans)로 씁니다.
커머스는 영어 상점 harbor 의 dev 질의 6개(의미 4 · 운송장 번호 · 취소 주문), 무역은 서식 이름 5개 + 문서번호 1개입니다.
계획의 앱 언어는 한국어로 두어, 질의 글에서 언어를 고르는지(`stanza::detect_query_lang`)도 함께 봅니다.

```bat
python make_plan.py ml12 --out ml12.jsonl          & rem 커머스 72건
python make_plan.py mlship12 --out mlship12.jsonl  & rem 무역 72건
set LOGIS_STANZA_DIR=<stanza-onnx-export>\onnx_final   & rem 없으면 앱 폴더 models\stanza
run_eval.bat 실험이름 plan ml12.jsonl ml12_out.jsonl ml12.log
python score_search.py ml12_out.jsonl --queries 12lang --by lang --seed-map seed_map.json --seeds seeds.jsonl
python score_search.py mlship12_out.jsonl 무역추출결과.jsonl --domain shipping --queries 12lang --by lang
```

검색 로그의 `[STANZA-POS-LOG]` · `[ANALYTIC] 🧠 Stanza` 줄이 언어 · 단위(공백 어절 / Stanza 토큰) · 어절별 품사 · 어간 · 관계 · MWT 낱말을 보여 줍니다.
12개 언어 사전이 필요하면 평가 폴더에 `bias.override.json`(full_bias.json 의 언어 구역)을 두거나 `--features full-bias` 로 빌드합니다.

평가 폴더에는 `models` 만 앱 데이터의 모델 폴더로 연결하고 DB · SDS 원장은 새로 쌓습니다. 앱 데이터(`%LOCALAPPDATA%\logis-center`)의 DB 와 원장은 바뀌지 않습니다. 모델은 앱의 다운로드 로직으로 받습니다. 언어 전용 4B 모델은 계획에 `{"id":"fetch","kind":"lang_engine","lang":"korean"}` 한 줄을 두면 앱의 `await_engine` 이 받습니다.

`run_eval.bat` 은 실행 중에 고치지 않습니다. cmd 는 배치 파일을 바이트 위치로 읽어서, 고치면 다음 줄을 잘못 읽습니다.

### 비전 게이트 (SigLIP2 경로, 합성 열화)

전체 이미지 추출(크롭마다 언어 모델)은 쪽당 5~25분이라, 읽은 값을 받을지 정하는 게이트만 실제 함수로 잽니다.

```bat
python rasterize_trading.py pages                                   & rem 영어 45건
python rasterize_trading.py pages_ml ..\trading\ml\output_pdfs      & rem 11개 언어 55건
python make_vision_cases.py pages pages_ml cases
run_eval.bat 실험이름 gates cases gates_block.jsonl gates.log block   & rem 출처 상자 = 카테고리 크롭 (line = 값 줄 ±1줄)
run_eval.bat 실험이름 names cases names_out.jsonl names.log
run_eval.bat 실험이름 labels cases labels_out.jsonl labels.log cpu
python score_vision_gates.py legibility gates_block.jsonl
python score_vision_gates.py names cases\names.jsonl names_out.jsonl
python score_vision_gates.py labels labels_out.jsonl
```

- `gates` 는 모델 가중치 없이 판독성 지도(build_legibility_map)와 verify_claims_v2 만 돕니다. SDS 기준선은 서식 코드마다 calib 쪽(열화 없음)으로
  쌓고, 평가 사례는 사례마다 따로 된 하위 스코프에서 판정해 calib 기준선만 봅니다.
- `labels` 는 granite 로 앱과 같은 라벨 뱅크를 세웁니다. `LOGIS_SYNTH_SEED_CONFUSION=1` 이면 라벨마다 1 · 2위 축 쌍을 혼동 사전에 미리 적는
  상한 측정입니다.
- 크롭 하한(분할 크롭)은 `vision` 모드(vision_probe)로 크롭 계획을 남기고 `score_vision.py` 로 정답 값이 크롭 안에 드는지 봅니다.
  `LOGIS_SYNTH_LIST` 로 쪽 순서(기준선을 쌓을 쪽 먼저)를 정합니다. `LOGIS_SYNTH_CPU=1` 은 CPU 로 돌지만 첫 쪽 구 인코딩만 수십 분이라 GPU 를 씁니다.

### 모델 정밀도 (PYRAMID-QUANT)

SigLIP2 는 계획(`LOGIS_SIGLIP2_QUANT`: `bf16` · `f32` · `q8` · `pyramid` · `<기준>;<묶음>=<형식>`)마다 새 평가 폴더에서 같은 쪽 순서로 돌리고
원본 F32 와 쌍으로 비교합니다. `LOGIS_SYNTH_HEAT=1` 이 카테고리 히트맵을 `<덤프>/<쪽>.heat.json` 으로 남깁니다.
`LOGIS_SIGLIP2_QUANT_SIM=1` 은 블록을 F32 로 풀어 올리는 측정 경로입니다(계획끼리 연산이 같아 가중치 값의 영향만 봅니다).

```bat
set "LOGIS_SYNTH_HEAT=1"
set "LOGIS_SIGLIP2_QUANT=f32" & run_eval.bat pq-f32 vision pages f32\result.jsonl f32\log.txt korean f32\dump
set "LOGIS_SIGLIP2_QUANT=q8"  & run_eval.bat pq-q8  vision pages q8\result.jsonl  q8\log.txt  korean q8\dump
python score_vision_quant.py pages f32=f32 q8=q8
```

언어 모델 쪽은 실모델 시험입니다(`LOGIS_TEST_MODELS=1 ... -- --ignored --nocapture --test-threads 1`, 시험 하나씩 돌릴 때
`LOGIS_TEST_FAST_EXIT=1` 이면 ROCm 종료 멈춤 없이 끝납니다): `p2b_qmatmul_parity` · `p2b_lm_head_cost` · `p2b_qwen35_2b_lm_head_q8` ·
`p2b_qwen35_4b_quantizer`(`LOGIS_LANG_QUANT_DIR` 에 변환본을 만듭니다).

RX 6600 (gfx1032) 실측, 2026-11-15:

| SigLIP2 (원본 F32 기준) | 영어 88쪽 ΔAP [95 %] | 문서번호 | 크롭 잃음 | 88쪽 시간 | 11개 언어 110쪽 ΔAP | F32 와 같은 서식 | 110쪽 시간 | VRAM 최대 |
|---|---|---|---|---|---|---|---|---|
| bf16 (예전 ROCm 기본) | −0.0008 [−0.0044, +0.0020] | 44/45 | 5 | 602 s | +0.0009 [−0.0014, +0.0034] | 39/55 | 688 s | 2,088 MB |
| **q8 (새 ROCm 기본)** | +0.0006 [−0.0005, +0.0017] | 45/45 | 2 | 538 s ¹ | +0.0008 [−0.0005, +0.0023] | 50/55 | 616 s | 1,438 MB |
| pyramid (지시문 4계층) | −0.0006 [−0.0036, +0.0020] | **43/45** | 6 | — | — | — | — | — |
| f32 | 0 | 45/45 | 0 | 461 s | 0 | — | 609 s | 3,954 MB |

¹ 타워 GGUF 를 처음 한 번 만드는 49 s 제외. 첫 쪽(구 1,230개 인코딩)은 bf16 134 s · q8 90 s · f32 69 s — 이 GPU 는 BF16 GEMM 이 F32 보다 느립니다.

- 4계층은 평균 AP 는 기준 안이지만 문서번호 크롭 두 개를 놓쳐(게이트 '문서번호 회수 100 %') 기각했습니다. q8 보다 아끼는 것은 비전 87 MB 입니다.
- 4304 는 32 의 배수도 아니라(32 × 134 + 16) 'Block=32 코덱으로 분기' 로는 안 되고, fc1 에 0 행을 붙여 fc2 입력을 4320 으로 냅니다.
- 커널 대조(`p2b_qmatmul_parity`): Q8_0 · Q6_K · Q5_K · Q4_K, 행 1 ~ 256 모두 ‖Δ‖/‖기준‖ ≤ 6.5e-7.
- Qwen3.5-2B 출력 헤드(묶인 임베딩): F16 역양자화본 → GGUF 의 Q8_0 그대로. 디코드 45.4 → 35.4 ms/토큰, 상주 −508 MB, teacher-forcing top-1 100 % ·
  KLD 7e-6, 탐욕 생성 3 × 96 토큰 같음 → ROCm 기본.
- 출력 헤드 자르기(16,384 행)의 상한: 한 토큰 0.6B Q8_0 3.5 → 0.3 ms, 2B Q8_0 8.4 → 0.5 ms. 12개 언어 값을 내는 모델이라 잘린 어휘의 오독을
  막을 정확한 방법(잘린 행 밖의 최댓값 보장)이 없어 넣지 않았습니다.
- 4B Q4_K_M 척도 탐색(Q8_0 기준): candle ΔNLL +0.032 · KLD 0.0455, 가중 최소제곱 ΔNLL +0.040 · KLD 0.0398 → 레시피 그대로.
- 검색 게이트(`SyntheticData/relational`, Phase 3 최종과 같은 SDS 스냅숏): 2B 생성을 타는 무역 · 행동 로그 질의의 5개 언어(ko · en · ja · ar · cs)
  부분집합 65개가 질의별로 같음(상위 10 결과 65/65, MRR 0.9736 그대로). 커머스 질의는 0.6B(바뀌지 않음)를 씁니다.

## 분할

튜닝은 dev 로만 합니다. test 는 개선을 고른 뒤 한 번 확인하는 데만 씁니다.

- 커머스: 쇼핑몰 4곳 dev, 2곳 test. 레이아웃이 겹치지 않습니다.
- 무역: Part 번호가 3의 배수인 15건 test, 나머지 30건 dev.
- ml: 한국어 · 영어 밖 언어의 합성 데이터. 커머스는 10개 언어 쇼핑몰 10곳(`commerce/lang_packs`), 무역은 11개 언어 × 서식 5종(`trading/ml`). 질의도 그 언어로 씁니다. 언어별로 보려면 `score_search.py --by lang`, 무역 추출은 `score_trading.py --split ml` 이 언어별 줄을 덧붙입니다.

```bat
python make_plan.py seeds --split ml --out seeds_ml.jsonl         & rem 커머스 정답 레코드 999건 (통화는 쇼핑몰 통화)
python make_plan.py search --split ml --out search_ml.jsonl       & rem 커머스 질의 648건
python make_plan.py trading --split ml --out trading_ml.jsonl     & rem 무역 서류 55건
python make_plan.py shipsearch --split ml --out ship_ml.jsonl     & rem 무역 질의 187건
```

값 비교(`normalize.py`)는 유럽식 숫자(`1.234,50` · `2 840,0` · `174,42`)와 날짜(`26.07.2026` · `26/07/2026` · `26. 7. 2026` · `2026年7月26日`)도 같은 값으로 봅니다. 영어 · 한국어 결과의 채점은 바뀌지 않습니다(무역 45건으로 대조).

분할 ml 무역 추출을 돌릴 때 알아 둘 것:

- 라틴 문자가 아닌 값(ja · ar · ko …)은 음차 별칭 색인 단계에서 디버그 테스트 실행기가 기본 스택(2 MB)을 넘겨 `0xC00000FD` 로 죽습니다. `set RUST_MIN_STACK=268435456` 을 두고 돌립니다. 앱 쪽 스택 설정은 없습니다(릴리스 빌드에서도 나는지는 확인하지 않았습니다).
- ja · zh 서류는 언어 전용 4B 음차 엔진(`Qwen3.5-4B-jpn` · `-zho`, 7.9 GB + 실행용 2.2 GB)이 없으면 앱이 평가 폴더의 `models`(앱 모델 폴더로 연결)로 받고, 받는 동안 그 문서에서 기다립니다. 디스크가 모자라면 그 언어 서류를 계획에서 뺍니다.
- 저장소(LanceDB)는 쓰기마다 버전 목록을 남겨, 시드 1000건이면 `item_chunks.lance/_versions` 가 수 GB 가 됩니다. 평가 폴더를 복사할 때 디스크를 넉넉히 둡니다.

검색 시드는 `make_plan.py seeds` 로 만듭니다. 주문 `no` · 운송장 `tracking_number` 칸까지 넣어야 식별자 조건을 공정하게 잽니다. 그 칸이 빠진 예전 시드로 잰 수치는 쓰지 않습니다.

합성 데이터는 내가 만든 것이라 숫자는 개선 전후 비교와 회귀 검사로 씁니다. 절댓값을 제품 성능으로 말하지 않습니다.
