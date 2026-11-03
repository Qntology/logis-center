# 합성 데이터 평가

합성 데이터를 실제 파이프라인에 태워 정답과 비교하고, 앱이 AppData 에 남긴 SDS 실행 파일과 교차 확인합니다. 브라우저 · 프론트엔드 · 창 없이 앱과 같은 `process_task` / `ai_search_complex` 를 부릅니다 (`commerce/src-tauri/tests/it/synthetic_eval.rs`).

## 파일

| 파일 | 역할 |
|---|---|
| `make_plan.py` | 평가 계획(JSONL). 커머스 페이지 · 무역 문서 · 페이지 이미지 · 검색 질의 · 정답 레코드 색인 |
| `run_eval.bat` | ROCm 테스트 바이너리 실행. 실험마다 `%LOCALAPPDATA%\logis-center-eval\<이름>` 을 씁니다 |
| `rasterize_trading.py` | 무역 PDF → 페이지 PNG + 단어 위치. 앱은 PDF 를 이미지로 바꾸지 않으므로 SigLIP2 경로를 재려면 필요합니다 |
| `score_pages.py` | 페이지 유형 · 목록/상세 판정 (실행 로그에서 읽음) |
| `score_commerce.py` · `score_trading.py` | 저장된 아이템과 정답 비교. `score_commerce.py --pool` 은 `{"id":"dump","kind":"dump"}` 계획으로 남긴 저장소 끝 상태로 찾습니다. 상세가 앞서 저장된 목록 문서에 합쳐지면 그 페이지의 결과에는 아이템이 남지 않기 때문입니다 |
| `score_vision.py` | SigLIP2 서식 판정 · 크롭 커버리지 |
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

평가 폴더에는 `models` 만 앱 데이터의 모델 폴더로 연결하고 DB · SDS 원장은 새로 쌓습니다. 앱 데이터(`%LOCALAPPDATA%\logis-center`)의 DB 와 원장은 바뀌지 않습니다. 모델은 앱의 다운로드 로직으로 받습니다. 언어 전용 4B 모델은 계획에 `{"id":"fetch","kind":"lang_engine","lang":"korean"}` 한 줄을 두면 앱의 `await_engine` 이 받습니다.

`run_eval.bat` 은 실행 중에 고치지 않습니다. cmd 는 배치 파일을 바이트 위치로 읽어서, 고치면 다음 줄을 잘못 읽습니다.

## 분할

튜닝은 dev 로만 합니다. test 는 개선을 고른 뒤 한 번 확인하는 데만 씁니다.

- 커머스: 쇼핑몰 4곳 dev, 2곳 test. 레이아웃이 겹치지 않습니다.
- 무역: Part 번호가 3의 배수인 15건 test, 나머지 30건 dev.

검색 시드는 `make_plan.py seeds` 로 만듭니다. 주문 `no` · 운송장 `tracking_number` 칸까지 넣어야 식별자 조건을 공정하게 잽니다. 그 칸이 빠진 예전 시드로 잰 수치는 쓰지 않습니다.

합성 데이터는 내가 만든 것이라 숫자는 개선 전후 비교와 회귀 검사로 씁니다. 절댓값을 제품 성능으로 말하지 않습니다.
