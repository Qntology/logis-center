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
set "VCVARS_ARG="
if defined VCVARS_VER set "VCVARS_ARG=-vcvars_ver=%VCVARS_VER%"
rem Per-call log folder: two consoles running this at once must not share one file.
call :unique_tmp VCVARS_DIR
set "VCVARS_LOG=%VCVARS_DIR%\vcvars_out.txt"
call "%VS_SCRIPT%" %VS_ARGS% %VCVARS_ARG% > "%VCVARS_LOG%" 2>&1
echo [MSVC] requested=%VCVARS_VER% active=%VCToolsVersion%
type "%VCVARS_LOG%"
del "%VCVARS_LOG%" > nul 2>&1
rmdir "%VCVARS_DIR%" > nul 2>&1
set "PYTHONIOENCODING=utf-8"

if not exist "%SystemRoot%\System32\vulkan-1.dll" (
    echo [Vulkan] vulkan-1.dll not found. It ships with the GPU driver; without it the app falls back to CPU.
)

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

:unique_tmp
rem Creates a fresh folder under %TEMP% and stores its path in the variable named by %1.
rem mkdir is atomic, so consoles that start in the same second (same %RANDOM% seed) cannot share it.
set "UT_DIR=%TEMP%\logis_env_%RANDOM%%RANDOM%"
mkdir "%UT_DIR%" > nul 2>&1 || goto :unique_tmp
set "%~1=%UT_DIR%"
set "UT_DIR="
exit /b 0
