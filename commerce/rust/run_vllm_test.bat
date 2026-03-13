@echo off
chcp 65001 > nul
echo [VLLM-ENV] Initializing Visual Studio 2019 Environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

set "PYTHONIOENCODING=utf-8"

echo [VLLM-TEST] Compiling and starting official vllm.rs example (rust-demo)...
cd vllm.rs-main\example\rust-demo

rem 명시적으로 CUDA 기능을 활성화합니다. 부모 폴더의 vllm-rs 라이브러리를 사용합니다.
cargo run --release --no-default-features --features cuda
