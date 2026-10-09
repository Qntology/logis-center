@echo off
cd /d "%~dp0"
rem Vulkan dev run, op profiler on, f32 compute dtype (A/B)
set "CANDLE_VULKAN_PROFILE=1"
set "LOGIS_VK_DTYPE=f32"
call "%~dp0dev_cargo_vulkan.bat" > dev_vulkan_prof_f32dtype.log 2>&1
