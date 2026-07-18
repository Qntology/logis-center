@echo off
chcp 65001 > nul
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "PYTHONIOENCODING=utf-8"

:: 2. CUDA 13.x 용 표준 전처리 가속 플래그 설정 (CCCL 빌드 에러 방지)
set "NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor"

:: [AMD ROCm/HIP]
set "CANDLE_HIP=1"
set "HIP_PATH=C:\Program Files\AMD\ROCm\7.1"
set "PATH=%HIP_PATH%\bin;%PATH%"

cd src-tauri

set "PATH=%CD%\dlls;%PATH%"

echo [DEV] Starting Tauri application (Development Mode)...

:: build 대신 dev 명령어를 사용하여 디버깅 모드로 앱을 실행합니다.
cargo tauri dev -- --features cuda