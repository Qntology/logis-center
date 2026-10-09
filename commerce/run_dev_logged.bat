@echo off
cd /d "%~dp0"
call "%~dp0dev_cargo_rocm.bat" > dev_rocm.log 2>&1
