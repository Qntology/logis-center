@echo off
rem =====================================================================
rem  CUDA build/test/run environment locator for Windows (MSVC + CUDA 13.x).
rem  Finds vcvars64.bat via VS BuildTools 18, VS 2022/2026, or vswhere.exe.
rem =====================================================================
set "PYTHONIOENCODING=utf-8"
set "NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor"

if defined VSCMD_VER goto :have_msvc

set "_VCVARS="
for %%P in (
    "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\18\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
) do (
    if not defined _VCVARS if exist %%~P set "_VCVARS=%%~P"
)

if not defined _VCVARS (
    set "_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    if exist "%_VSWHERE%" (
        for /f "usebackq delims=" %%I in (`"%_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
            if exist "%%I\VC\Auxiliary\Build\vcvars64.bat" set "_VCVARS=%%I\VC\Auxiliary\Build\vcvars64.bat"
        )
    )
)

if not defined _VCVARS (
    echo [ERROR] Could not locate Visual Studio vcvars64.bat for CUDA build.
    exit /b 1
)

call "%_VCVARS%" > nul
if errorlevel 1 exit /b 1

:have_msvc
if not defined NVCC_CCBIN (
    for /f "delims=" %%C in ('where cl.exe 2^>nul') do if not defined NVCC_CCBIN set "NVCC_CCBIN=%%C"
)
exit /b 0
