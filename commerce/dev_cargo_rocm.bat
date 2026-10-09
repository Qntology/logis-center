@echo off
chcp 65001 > nul
call "%~dp0_env_rocm.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0src-tauri"
set "PATH=%CD%\dlls;%PATH%"

echo [DEV] Starting Tauri application (ROCm / HIP %ROCM_VER%: %ROCM_PATH%)...
call %TAURI_CLI% dev -- --no-default-features --features rocm
