@echo off
cd /d "%~dp0"
rem Vulkan dev run with the op profiler on and the F16 embedding table (A/B against the Q8 table)
set "CANDLE_VULKAN_PROFILE=1"
set "LOGIS_EMBED_Q8=0"
call "%~dp0dev_cargo_vulkan.bat" > dev_vulkan_f16embed_prof.log 2>&1
