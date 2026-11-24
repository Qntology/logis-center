@echo off
setlocal
chcp 65001 > nul
net session >nul 2>&1
if errorlevel 1 (
    echo Requesting administrator rights...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)
set "VSSETUP=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\setup.exe"
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSROOT="
if exist "%VSWHERE%" (
    for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -property installationPath`) do if not defined VSROOT set "VSROOT=%%I"
)
if not defined VSROOT set "VSROOT=%ProgramFiles%\Microsoft Visual Studio\2022\Community"
if not exist "%VSSETUP%" (
    echo [ERROR] Visual Studio Installer not found: "%VSSETUP%"
    pause
    exit /b 1
)
if not exist "%VSROOT%" (
    echo [ERROR] Visual Studio installation not found: "%VSROOT%"
    pause
    exit /b 1
)
echo Installing Windows 11 SDK 26100 into "%VSROOT%" ...
"%VSSETUP%" modify --installPath "%VSROOT%" --add Microsoft.VisualStudio.Component.Windows11SDK.26100 --passive --norestart --force
set "RC=%errorlevel%"
echo Exit code: %RC%
if exist "%ProgramFiles(x86)%\Windows Kits\10\Lib" (echo OK: Windows SDK installed.) else (echo FAILED: Windows SDK Lib folder still missing.)
pause
exit /b %RC%
