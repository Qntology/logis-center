@echo off
chcp 65001 > nul
call "%~dp0_env_vulkan.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0src-tauri"

cargo check --no-default-features --features vulkan
