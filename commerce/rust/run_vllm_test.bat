@echo off
setlocal
chcp 65001 > nul
echo [VLLM-ENV] Initializing Visual Studio 2019 Environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

set "PYTHONIOENCODING=utf-8"

echo [VLLM-TEST] Compiling and starting official vllm.rs example (rust-demo)...
cd /d "%~dp0vllm.rs-main\example\rust-demo"

echo [VLLM-TEST] Current directory: %cd%
if not exist Cargo.toml (
    echo Error: Cargo.toml not found in current directory.
    exit /b 1
)

echo [VLLM-TEST] Running cargo run...
cargo run --release --no-default-features --features cuda
if errorlevel 1 (
    echo Error: cargo run failed.
    exit /b 1
)

endlocal
