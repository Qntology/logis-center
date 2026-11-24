@echo off
setlocal
chcp 65001 > nul
call "%~dp0_env_vulkan.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0..\apps\desktop"

echo [BUILD] Compiling and bundling Tauri application (Vulkan)...
call %TAURI_CLI% build %* -- --no-default-features --features vulkan
exit /b %errorlevel%
