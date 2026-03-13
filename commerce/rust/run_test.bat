@echo off
chcp 65001 > nul
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "PYTHONIOENCODING=utf-8"
cd src-tauri

rem Add DirectStorage DLLs to PATH
set "PATH=%PATH%;%CD%\microsoft.direct3d.directstorage.1.3.0\native\bin\x64"

echo [TEST] Compiling and starting Qwen3.5 Test (UTF-8 Mode)...
cargo run --bin test_qwen
