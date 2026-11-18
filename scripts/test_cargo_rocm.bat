@echo off
chcp 65001 > nul
call "%~dp0_env_rocm.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0..\apps\desktop"
set "PATH=%CD%\dlls;%PATH%"

rem 통합 시험은 apps/core/tests/it (logis-app, 시험 exe 에 Tauri 없음). -p logis-center 는 앱과 같은 기능 집합으로 빌드하게 합니다.
echo [TEST] cargo test -p logis-center -p logis-app --test it (ROCm / HIP %ROCM_VER%: %ROCM_PATH%)...
cargo test -p logis-center -p logis-app --test it --no-default-features --features rocm %*
