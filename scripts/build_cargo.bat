@echo off
chcp 65001 > nul
call "%~dp0_env_cuda.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0..\apps\desktop"
set "TAURI_APP_PATH=%~dp0..\apps\desktop"
set "TAURI_FRONTEND_PATH=%~dp0..\web\desktop"

set "PATH=%CD%\dlls;%PATH%"

echo [BUILD] Compiling and bundling Tauri application (Release and UTF-8 Mode)...

:: cargo build 대신 cargo tauri build를 사용하여 .msi 번들링까지 수행합니다.
:: 기존 컴파일 캐시를 재사용하기 위해 -- --features cuda 옵션을 그대로 유지합니다.
cargo tauri build -- --features cuda