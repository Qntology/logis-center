@echo off
setlocal
chcp 65001 > nul
call "%~dp0_env_vulkan.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0..\apps\desktop"

echo [DEV] Starting Tauri application (Vulkan)...
call %TAURI_CLI% dev %* -- --no-default-features --features vulkan
exit /b %errorlevel%
