@echo off

set "VS_SCRIPT="
set "VS_ARGS="
set "VS2022=%ProgramFiles%\Microsoft Visual Studio\2022"
for %%E in (Community Professional Enterprise BuildTools) do call :probe_vs "%VS2022%\%%E"
if not defined VS_SCRIPT call :probe_vswhere
if not defined VS_SCRIPT (
    echo [MSVC] Visual Studio C++ Build Tools not found. Install the "Desktop development with C++" workload.
    exit /b 1
)
if not defined VCVARS_VER set "VCVARS_VER=14.44"
call "%VS_SCRIPT%" %VS_ARGS% -vcvars_ver=%VCVARS_VER% > "%TEMP%\vcvars_out.txt" 2>&1
echo [MSVC] requested=%VCVARS_VER% active=%VCToolsVersion%
type "%TEMP%\vcvars_out.txt"
set "PYTHONIOENCODING=utf-8"

rem ---------------------------------------------------------------------------------------------
rem ROCm: two tracks side by side
rem   legacy : AMD HIP SDK 7.x            C:\Program Files\AMD\ROCm\7.2
rem   core   : ROCm Core SDK 10.x         tarball extracted to C:\Program Files\AMD\ROCm\10.1
rem                                       (or anywhere listed in ROCM_SEARCH_PATHS), or rocm-sdk (pip)
rem Choose with (first match wins):
rem   ROCM_VER=7.2 / 10.1          a folder under C:\Program Files\AMD\ROCm (core-10.1 is also tried)
rem   ROCM_PATH=<root>             any installation
rem   CANDLE_ROCM_TRACK=auto       (default) core when every GPU of this PC is supported by ROCm 10.1,
rem                                otherwise legacy. legacy / core force a track.
rem ---------------------------------------------------------------------------------------------

rem Values set by a previous run in this console are not user choices.
if defined ROCM_ENV_AUTO_VER if "%ROCM_VER%"=="%ROCM_ENV_AUTO_VER%" set "ROCM_VER="
if defined ROCM_ENV_AUTO_PATH if "%ROCM_PATH%"=="%ROCM_ENV_AUTO_PATH%" set "ROCM_PATH="
set "ROCM_ENV_AUTO_VER="
set "ROCM_ENV_AUTO_PATH="
set "ROCM_TRACK="
set "ROCM_HIP_VER="
set "ROCM_SEL_NOTE="

if defined ROCM_VER goto :rocm_from_ver
if defined ROCM_PATH goto :rocm_from_path
goto :rocm_auto

:rocm_from_ver
set "ROCM_PATH=%ProgramFiles%\AMD\ROCm\%ROCM_VER%"
if not exist "%ROCM_PATH%\bin" if exist "%ProgramFiles%\AMD\ROCm\core-%ROCM_VER%\bin" set "ROCM_PATH=%ProgramFiles%\AMD\ROCm\core-%ROCM_VER%"
set "ROCM_ENV_AUTO_PATH=%ROCM_PATH%"
goto :rocm_check

:rocm_from_path
if "%ROCM_PATH:~-1%"=="\" set "ROCM_PATH=%ROCM_PATH:~0,-1%"
for %%I in ("%ROCM_PATH%") do set "ROCM_VER=%%~nxI"
set "ROCM_ENV_AUTO_VER=%ROCM_VER%"
goto :rocm_check

:rocm_auto
set "ROCM_SEL_ROOT="
set "ROCM_SEL_VER="
set "ROCM_SEL_TRACK="
set "ROCM_SEL_HIP="
set "ROCM_SEL_ARCHS="
for /f "usebackq tokens=1,* delims==" %%A in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0_rocm_select.ps1"`) do set "ROCM_SEL_%%A=%%B"
if not defined ROCM_SEL_ROOT goto :rocm_not_found
set "ROCM_PATH=%ROCM_SEL_ROOT%"
set "ROCM_VER=%ROCM_SEL_VER%"
set "ROCM_TRACK=%ROCM_SEL_TRACK%"
set "ROCM_HIP_VER=HIP %ROCM_SEL_HIP%"
set "ROCM_ENV_AUTO_VER=%ROCM_VER%"
set "ROCM_ENV_AUTO_PATH=%ROCM_PATH%"
if defined ROCM_SEL_NOTE echo [ROCm] %ROCM_SEL_NOTE%
goto :rocm_check

:rocm_not_found
echo [ROCm] No usable ROCm / HIP SDK installation found.
if defined ROCM_SEL_NOTE echo [ROCm] %ROCM_SEL_NOTE%
echo        Installed under "%ProgramFiles%\AMD\ROCm":
dir /b /ad "%ProgramFiles%\AMD\ROCm" 2>nul
echo        Set ROCM_VER, ROCM_PATH or ROCM_SEARCH_PATHS.
exit /b 1

:rocm_check
if not exist "%ROCM_PATH%\bin\hipcc.exe" if not exist "%ROCM_PATH%\lib\llvm\bin\clang++.exe" (
    echo [ROCm] HIP SDK %ROCM_VER% not found: %ROCM_PATH%
    echo        Installed versions:
    dir /b /ad "%ProgramFiles%\AMD\ROCm" 2>nul
    exit /b 1
)
if not defined ROCM_TRACK (
    set "ROCM_TRACK=legacy"
    findstr /r /c:"define HIP_VERSION_MINOR  *[1-9][0-9]" "%ROCM_PATH%\include\hip\hip_version.h" >nul 2>&1 && set "ROCM_TRACK=core"
)
set "HIP_PATH=%ROCM_PATH%"
set "HIPCC="
if exist "%ROCM_PATH%\bin\hipcc.exe" set "HIPCC=%ROCM_PATH%\bin\hipcc.exe"
set "PATH=%ROCM_PATH%\bin;%PATH%"
if exist "%ROCM_PATH%\lib\llvm\bin" set "PATH=%PATH%;%ROCM_PATH%\lib\llvm\bin"
echo [ROCm] %ROCM_TRACK% track: %ROCM_PATH% %ROCM_HIP_VER%

if not exist "%~dp0src-tauri\dlls" mkdir "%~dp0src-tauri\dlls"
if not exist "%~dp0src-tauri\dlls\.gitkeep" type nul > "%~dp0src-tauri\dlls\.gitkeep"

set "PATH=%~dp0src-tauri\microsoft.direct3d.directstorage.1.3.0\native\bin\x64;%PATH%"

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
