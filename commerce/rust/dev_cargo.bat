@echo off
chcp 65001 > nul
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "NVCC_CCBIN=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe"
set "PYTHONIOENCODING=utf-8"

:: 2. CUDA 13.x 용 표준 전처리 가속 플래그 설정 (CCCL 빌드 에러 방지)
set "NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor"

:: [AMD ROCm/HIP]
:: set "CANDLE_HIP=1"
:: set "HIP_PATH=C:\Program Files\AMD\ROCm\7.1"
:: set "PATH=%HIP_PATH%\bin;%PATH%"

cd src-tauri

set "CUDA_PATH=%CD%\dlls\cuda"
set "CUDA_ROOT=%CD%\dlls\cuda"
set "CUDA_TOOLKIT_ROOT_DIR=%CD%\dlls\cuda"
set "MSVC_BIN=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64"
set "PATH=%MSVC_BIN%;%CD%\dlls;%CD%\dlls\cuda\bin;%CD%\dlls\cuda\lib;%CD%\dlls\cuda\include;%PATH%"

echo [DEV] Starting Tauri application (Development Mode)...

:: build 대신 dev 명령어를 사용하여 디버깅 모드로 앱을 실행합니다.
cargo tauri dev -- --features cuda