@echo off
chcp 65001 > nul
call "%~dp0_env_cuda.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0..\apps\desktop"
set "TAURI_APP_PATH=%~dp0..\apps\desktop"
set "TAURI_FRONTEND_PATH=%~dp0..\web\desktop"

set "PATH=%CD%\dlls;%PATH%"

echo [DEV] Starting Tauri application (Development Mode)...

:: build 대신 dev 명령어를 사용하여 디버깅 모드로 앱을 실행합니다.
cargo tauri dev -- --features cuda