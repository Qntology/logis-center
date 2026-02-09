@echo off
chcp 65001 > nul
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "NVCC_CCBIN=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe"
set "PYTHONIOENCODING=utf-8"
cd src-tauri
echo [RUN] Compiling and starting application (UTF-8 Mode)...
cargo run
