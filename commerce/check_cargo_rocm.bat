@echo off
chcp 65001 > nul
call "%~dp0_env_rocm.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0src-tauri"
set "PATH=%CD%\dlls;%PATH%"

echo [CHECK] cargo check (ROCm / HIP %ROCM_VER%: %ROCM_PATH%)...
cargo check --no-default-features --features rocm
