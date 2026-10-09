@echo off
chcp 65001 > nul
rem Installs the Windows 11 SDK (kernel32.lib etc.) into VS 2022 Community. Needs admin (UAC prompt).
net session >nul 2>&1
if errorlevel 1 (
    echo Requesting administrator rights...
    powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)
set "VSROOT=%ProgramFiles%\Microsoft Visual Studio\2022\Community"
set "VSSETUP=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\setup.exe"
echo Installing Windows 11 SDK 26100 into "%VSROOT%" ...
"%VSSETUP%" modify --installPath "%VSROOT%" --add Microsoft.VisualStudio.Component.Windows11SDK.26100 --passive --norestart --force
echo Exit code: %errorlevel%
if exist "%ProgramFiles(x86)%\Windows Kits\10\Lib" (echo OK: Windows SDK installed.) else (echo FAILED: Windows SDK Lib folder still missing.)
pause
