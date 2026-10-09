@echo off
chcp 65001 > nul
call "%~dp0_env_rocm.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0src-tauri"
set "PATH=%CD%\dlls;%PATH%"

echo [BUILD] Compiling and bundling Tauri application (ROCm / HIP %ROCM_VER%: %ROCM_PATH%)...
call %TAURI_CLI% build -- --no-default-features --features rocm
