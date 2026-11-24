@echo off
setlocal
chcp 65001 > nul
call "%~dp0_env_rocm.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0..\apps\desktop"

echo [CHECK] cargo check (ROCm / HIP %ROCM_VER%: %ROCM_PATH%)...
cargo check --no-default-features --features rocm %*
exit /b %errorlevel%
