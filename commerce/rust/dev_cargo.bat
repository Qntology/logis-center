@echo off
chcp 65001 > nul
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "PYTHONIOENCODING=utf-8"

:: [AMD ROCm/HIP]
set "CANDLE_HIP=1"
set "HIP_PATH=C:\Program Files\AMD\ROCm\7.1"
set "PATH=%HIP_PATH%\bin;%PATH%"

cd src-tauri

rem DirectStorage 및 아까 모아둔 CUDA DLL 경로를 임시로 PATH에 추가 (Dev 모드 실행을 위함)
set "PATH=%PATH%;%CD%\microsoft.direct3d.directstorage.1.3.0\native\bin\x64"
set "PATH=%PATH%;%CD%\dlls"

echo [DEV] Starting Tauri application (Development Mode)...

:: build 대신 dev 명령어를 사용하여 디버깅 모드로 앱을 실행합니다.
cargo tauri dev -- --features cuda