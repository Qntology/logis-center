@echo off
chcp 65001 > nul
call "%~dp0_env_cuda.bat"
if errorlevel 1 exit /b 1

cd /d "%~dp0..\apps\desktop"

rem Add DirectStorage DLLs to PATH
set "PATH=%PATH%;%CD%\microsoft.direct3d.directstorage.1.3.0\native\bin\x64"

echo [RUN] Compiling and starting application (CUDA / UTF-8 Mode)...
cargo run --features cuda %*
