@echo off
chcp 65001 > nul
call "%~dp0_env_rocm.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0src-tauri"
set "PATH=%CD%\dlls;%PATH%"

echo [TEST] cargo test --test it (ROCm / HIP %ROCM_VER%: %ROCM_PATH%)...
cargo test --test it --no-default-features --features rocm %*
