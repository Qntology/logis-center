$env:NVCC_CCBIN = "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe"; cargo check

이거 먼저 선언후에

legacy_python_code 폴더 안 소스가 일부 누락되었어 반영해줘
- app.py 
- logic.py
- search_engine.py

그리고 프론트 소스는 참고 소스인 python에서 gradio를 기반으로 작성해야되 

수정한 다음에 문법 검색부터 먼저 하고 완료가 되면
legacy_python_code 폴더 안 소스가 Rust 프로젝트에 참고 및 반영이 안된 상태에서 절대 다음 단계로 넘어가지마




그리고 이전 cargo run 과정에서 Visual Studio 2019로 생성된 빌드 아티팩트가 target 폴더에 남아있는데, 

1. cargo clean
2. cargo check 한다음에

 시스템 기본값인 VS 2022가 이를 덮어쓰려다 충돌이 되지 않게 검증해 그리고 완료 되면 cargo run 진행해줘




지시 (문법 교정 + 로그 추적)

"방금 시도한 `{{ }}` 수정은 명백한 문법 오류야. 너는 지금 중괄호 이스케이프를 잘못 이해하고 있어. 아래 지시를 엄격히 따라."

**1. 중괄호(`{}`) 사용 규칙 (필독):**

* **`json!` 매크로 내부:** 무조건 `{ "key": "value" }` 처럼 **중괄호 하나**만 사용해. `{{`를 쓰면 Rust 컴파일러가 인식하지 못해.
* **`println!`, `format!` 내부:** 변수가 들어갈 자리에는 `{}` 하나만 써. (예: `println!("{}", value);`)
* **파일 전체 교정:** `src-tauri/src/model.rs`에서 네가 방금 넣은 `{{`들을 전부 정상적인 Rust 문법(단일 `{`)으로 되돌려놔.

**2. 실행 로그 모니터링 (가장 중요):**

* 빌드에 성공하면 앱이 켜질 때까지 기다려.
* 특히 **`[IMAGE-STATS]`**와 **`[DEBUG: RAW RESPONSE]`** 로그가 터미널에 찍히는지 **최소 1분 동안** 눈을 떼지 말고 지켜봐.
* 만약 로그가 터미널에 바로 안 보인다면, `cargo run > run_log.txt 2>&1` 명령어로 출력을 파일로 빼서라도 그 내용을 반드시 읽고 나에게 보고해.

**3. 실행 명령어 (환경 변수 포함):**

cmd /c "call `"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat`" && set `"NVCC_CCBIN=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe`" && cargo run --release"








지시 사항 (CMake 캐시 충돌 및 VS 버전 불일치 해결):"

현재 aws-lc-sys 빌드 중 Visual Studio 17 2022와 16 2019 간의 Generator Mismatch 에러가 발생했어. 이건 cargo clean으로 지워지지 않는 CMake 캐시 때문이야.

먼저, 프로젝트 루트에 있는 target 디렉토리를 완전히(물리적으로) 삭제해줘. (PowerShell의 Remove-Item -Recurse -Force target 등을 사용)

그 다음, 아래의 합체 명령어를 실행해서 VS 2019 환경으로 처음부터 다시 빌드(Fresh Build)를 시작해.

PowerShell

cmd /c "call `"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat`" && set `"NVCC_CCBIN=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe`" && cargo run -j 1"
CMake Error 없이 Compiling 로그가 처음부터 다시 올라오는지 모니터링해줘.