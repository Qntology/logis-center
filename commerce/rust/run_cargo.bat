@echo off
chcp 65001 > nul
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "PYTHONIOENCODING=utf-8"

:: 2. CUDA 13.x 용 표준 전처리 가속 플래그 설정 (CCCL 빌드 에러 방지)
set "NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor"

:: [AMD ROCm/HIP]
set "CANDLE_HIP=1"
set "HIP_PATH=C:\Program Files\AMD\ROCm\7.1"
set "PATH=%HIP_PATH%\bin;%PATH%"

cd src-tauri

rem Add DirectStorage DLLs to PATH
set "PATH=%PATH%;%CD%\microsoft.direct3d.directstorage.1.3.0\native\bin\x64"

echo [RUN] Compiling and starting application (UTF-8 Mode)...
cargo run --features cuda
