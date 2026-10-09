@echo off
chcp 65001 > nul
set "DEST=%~dp0src-tauri\microsoft.direct3d.directstorage.1.3.0"
set "LOG=%~dp0setup_directstorage.log"
echo start > "%LOG%"
if exist "%DEST%\native\bin\x64\dstorage.dll" (
    echo already present >> "%LOG%"
    exit /b 0
)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $z=Join-Path $env:TEMP 'directstorage.zip'; Invoke-WebRequest -UseBasicParsing -Uri 'https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.directstorage/1.3.0/microsoft.direct3d.directstorage.1.3.0.nupkg' -OutFile $z; if (Test-Path '%DEST%') { Remove-Item -Recurse -Force '%DEST%' }; Expand-Archive -Path $z -DestinationPath '%DEST%' -Force; Remove-Item $z" >> "%LOG%" 2>&1
if exist "%DEST%\native\bin\x64\dstorage.dll" (echo OK >> "%LOG%") else (echo FAILED >> "%LOG%")
