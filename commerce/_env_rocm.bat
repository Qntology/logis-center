@echo off
rem ============================================================
rem  Common ROCm / HIP environment (called by dev/build/check _cargo_rocm.bat)
rem  - Default HIP SDK version: 7.2   (override: set ROCM_VER=7.1 before running)
rem  - Overrides a system-wide HIP_PATH that still points to 7.1
rem ============================================================

rem ---- 1) MSVC environment ----
set "VS_SCRIPT="
set "VS_ARGS="
set "VS2022=%ProgramFiles%\Microsoft Visual Studio\2022"
for %%E in (Community Professional Enterprise BuildTools) do call :probe_vs "%VS2022%\%%E"
if not defined VS_SCRIPT call :probe_vswhere
if not defined VS_SCRIPT (
    echo [MSVC] Visual Studio C++ Build Tools not found. Install the "Desktop development with C++" workload.
    exit /b 1
)
rem HIP clang 21 (ROCm 7.2) cannot compile against the MSVC 14.51 STL (VS 2026 toolset): use 14.44 instead.
if not defined VCVARS_VER set "VCVARS_VER=14.44"
call "%VS_SCRIPT%" %VS_ARGS% -vcvars_ver=%VCVARS_VER% > "%TEMP%\vcvars_out.txt" 2>&1
echo [MSVC] requested=%VCVARS_VER% active=%VCToolsVersion%
type "%TEMP%\vcvars_out.txt"
set "PYTHONIOENCODING=utf-8"

rem ---- 2) ROCm / HIP SDK ----
if not defined ROCM_VER set "ROCM_VER=7.2"
set "ROCM_PATH=%ProgramFiles%\AMD\ROCm\%ROCM_VER%"
if not exist "%ROCM_PATH%\bin\hipcc.exe" (
    echo [ROCm] HIP SDK %ROCM_VER% not found: %ROCM_PATH%
    echo        Installed versions:
    dir /b /ad "%ProgramFiles%\AMD\ROCm" 2>nul
    exit /b 1
)
rem candle-rocm build scripts read ROCM_PATH first, then HIP_PATH. Pin both.
set "HIP_PATH=%ROCM_PATH%"
set "HIPCC=%ROCM_PATH%\bin\hipcc.exe"
set "PATH=%ROCM_PATH%\bin;%PATH%"

rem ---- 3) tauri.conf.json bundles "dlls/*"; the folder must exist or tauri-build fails ----
if not exist "%~dp0src-tauri\dlls" mkdir "%~dp0src-tauri\dlls"
if not exist "%~dp0src-tauri\dlls\.gitkeep" type nul > "%~dp0src-tauri\dlls\.gitkeep"

rem ---- 3b) DirectStorage runtime (dstorage.dll), fetched by setup_directstorage.bat ----
set "PATH=%~dp0src-tauri\microsoft.direct3d.directstorage.1.3.0\native\bin\x64;%PATH%"

rem ---- 4) Tauri CLI: prefer the local one from node_modules (cargo-tauri is not installed) ----
set "TAURI_CLI=cargo tauri"
if exist "%~dp0node_modules\.bin\tauri.cmd" set TAURI_CLI="%~dp0node_modules\.bin\tauri.cmd"
exit /b 0

:probe_vs
if defined VS_SCRIPT exit /b 0
if exist "%~1\VC\Auxiliary\Build\vcvars64.bat" (
    set "VS_SCRIPT=%~1\VC\Auxiliary\Build\vcvars64.bat"
    exit /b 0
)
if exist "%~1\Common7\Tools\VsDevCmd.bat" (
    set "VS_SCRIPT=%~1\Common7\Tools\VsDevCmd.bat"
    set "VS_ARGS=-arch=x64 -host_arch=x64"
)
exit /b 0

:probe_vswhere
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" exit /b 0
for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do call :probe_vs "%%I"
exit /b 0
