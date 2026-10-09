@echo off
cd /d "%~dp0"
call "%~dp0check_cargo_vulkan.bat" > check_vulkan.log 2>&1
echo CHECK_EXIT=%ERRORLEVEL% >> check_vulkan.log
