@echo off
cd /d "%~dp0"
rem Repeat of the default Vulkan run (determinism check)
set "CANDLE_VULKAN_PROFILE=1"
call "%~dp0dev_cargo_vulkan.bat" > dev_vulkan_prof_repeat.log 2>&1
