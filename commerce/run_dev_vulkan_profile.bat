@echo off
cd /d "%~dp0"
rem Vulkan dev run with the op profiler on (default settings otherwise)
set "CANDLE_VULKAN_PROFILE=1"
call "%~dp0dev_cargo_vulkan.bat" > dev_vulkan_prof.log 2>&1
