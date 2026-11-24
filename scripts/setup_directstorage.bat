@echo off
setlocal
chcp 65001 > nul
set "DEST=%~dp0..\apps\desktop\microsoft.direct3d.directstorage.1.3.0"
set "LOG=%~dp0setup_directstorage.log"
echo start > "%LOG%"
if exist "%DEST%\native\bin\x64\dstorage.dll" (
    echo [DirectStorage] already present: %DEST%\native\bin\x64\dstorage.dll
    echo already present >> "%LOG%"
    exit /b 0
)
echo [DirectStorage] Downloading Microsoft.Direct3D.DirectStorage 1.3.0 runtime DLLs...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $z=Join-Path $env:TEMP ('directstorage_' + $PID + '.zip'); $tmp=Join-Path $env:TEMP ('directstorage_' + $PID); try { Invoke-WebRequest -UseBasicParsing -Uri 'https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.directstorage/1.3.0/microsoft.direct3d.directstorage.1.3.0.nupkg' -OutFile $z; Expand-Archive -Path $z -DestinationPath $tmp -Force; $srcBin=Join-Path $tmp 'native\bin'; $dstBin=Join-Path '%DEST%' 'native\bin'; if (!(Test-Path $srcBin)) { throw 'native\bin missing in package' }; New-Item -ItemType Directory -Path $dstBin -Force | Out-Null; Copy-Item -Path (Join-Path $srcBin '*') -Destination $dstBin -Recurse -Force } finally { if (Test-Path $z) { Remove-Item -Force $z }; if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp } }" >> "%LOG%" 2>&1
if exist "%DEST%\native\bin\x64\dstorage.dll" (
    echo [DirectStorage] OK: %DEST%\native\bin\x64\dstorage.dll
    echo OK >> "%LOG%"
    exit /b 0
) else (
    echo [DirectStorage] FAILED. See %LOG%
    type "%LOG%"
    echo FAILED >> "%LOG%"
    exit /b 1
)
