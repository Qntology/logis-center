@echo off
setlocal
chcp 65001 > nul
call "%~dp0_env_cuda.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0..\apps\desktop"

echo [DEV] Starting Tauri application (CUDA / Development Mode)...

:: build 대신 dev 명령어를 사용하여 디버깅 모드로 앱을 실행합니다.
:: Cargo.toml 기본 features(cuda, flash-attn, turbo-quant)가 자동으로 활성화됩니다.
call %TAURI_CLI% dev %*
exit /b %errorlevel%