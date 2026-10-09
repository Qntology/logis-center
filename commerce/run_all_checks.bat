@echo off
cd /d "%~dp0"
call "%~dp0run_check_vulkan_logged.bat"
call "%~dp0run_check_logged.bat"
