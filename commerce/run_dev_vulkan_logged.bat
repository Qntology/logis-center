@echo off
cd /d "%~dp0"
call "%~dp0dev_cargo_vulkan.bat" > dev_vulkan.log 2>&1
