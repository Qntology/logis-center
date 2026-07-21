@echo off
chcp 65001 > nul
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "NVCC_CCBIN=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe"
set "PYTHONIOENCODING=utf-8"

:: 2. CUDA 13.x 용 표준 전처리 가속 플래그 설정 (CCCL 빌드 에러 방지)
set "NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor"

cd src-tauri

set "PATH=%CD%\dlls;%PATH%"

echo [BUILD] Compiling and bundling Tauri application (Release and UTF-8 Mode)...

:: cargo build 대신 cargo tauri build를 사용하여 .msi 번들링까지 수행합니다.
:: 기존 컴파일 캐시를 재사용하기 위해 -- --features cuda 옵션을 그대로 유지합니다.
cargo tauri build -- --features cuda