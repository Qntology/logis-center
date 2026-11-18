@echo off
chcp 65001 > nul
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "PYTHONIOENCODING=utf-8"

:: CUDA 13.x 용 표준 전처리 가속 플래그 설정 (CCCL 빌드 에러 방지)
set "NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor"
cd /d "%~dp0..\apps\desktop"

rem 통합 시험은 apps/core/tests/it (logis-app, 시험 exe 에 Tauri 없음). -p logis-center 는 앱과 같은 기능 집합으로 빌드하게 합니다.
echo [TEST] cargo test -p logis-center -p logis-app --test it (CUDA)...
cargo test -p logis-center -p logis-app --test it %*
