@echo off
chcp 65001 > nul
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "PYTHONIOENCODING=utf-8"

:: 2. CUDA 13.x 용 표준 전처리 가속 플래그 설정 (CCCL 빌드 에러 방지)
set "NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor"
cd src-tauri
cargo check
