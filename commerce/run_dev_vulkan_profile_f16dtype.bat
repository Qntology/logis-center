@echo off
cd /d "%~dp0"
rem Vulkan dev run with the op profiler on, f16 compute dtype (A/B against the default f32)
set "CANDLE_VULKAN_PROFILE=1"
set "LOGIS_VK_DTYPE=f16"
call "%~dp0dev_cargo_vulkan.bat" > dev_vulkan_prof_f16dtype.log 2>&1
