@echo off
chcp 65001 > nul
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "PYTHONIOENCODING=utf-8"
cd src-tauri
echo [RUN] Compiling and starting application (UTF-8 Mode)...
cargo run
