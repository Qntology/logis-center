@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo [1/2] npm install > verify_rocm.log
call npm install >> verify_rocm.log 2>&1
echo npm exit: %errorlevel% >> verify_rocm.log
echo [2/2] cargo check rocm >> verify_rocm.log
call "%~dp0check_cargo_rocm.bat" >> verify_rocm.log 2>&1
echo cargo exit: %errorlevel% >> verify_rocm.log
echo DONE >> verify_rocm.log
