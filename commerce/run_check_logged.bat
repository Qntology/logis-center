@echo off
cd /d "%~dp0"
call "%~dp0check_cargo_rocm.bat" > check_rocm.log 2>&1
echo CHECK_EXIT=%ERRORLEVEL% >> check_rocm.log
