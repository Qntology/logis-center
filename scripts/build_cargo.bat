@echo off
setlocal
chcp 65001 > nul
call "%~dp0_env_cuda.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0..\apps\desktop"

echo [BUILD] Compiling and bundling Tauri application (CUDA / Release Mode)...

:: cargo build 대신 tauri build를 사용하여 .msi/.exe 번들링까지 수행합니다.
:: Cargo.toml 기본 features(cuda, flash-attn, turbo-quant)가 자동으로 활성화됩니다.
call %TAURI_CLI% build %*
exit /b %errorlevel%