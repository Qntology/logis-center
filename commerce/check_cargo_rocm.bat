@echo off
chcp 65001 > nul
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "PYTHONIOENCODING=utf-8"

if not defined HIP_PATH (
    for /f "delims=" %%D in ('dir /b /ad /o-n "C:\Program Files\AMD\ROCm" 2^>nul') do (
        if not defined HIP_PATH set "HIP_PATH=C:\Program Files\AMD\ROCm\%%D\"
    )
)
if not defined HIP_PATH (
    echo [ROCm] AMD HIP SDK 를 찾을 수 없습니다. HIP SDK for Windows 설치 후 다시 실행하세요.
    exit /b 1
)
if not "%HIP_PATH:~-1%"=="\" set "HIP_PATH=%HIP_PATH%\"

cd src-tauri

set "PATH=%CD%\dlls;%PATH%;%HIP_PATH%bin"

cargo check --no-default-features --features rocm