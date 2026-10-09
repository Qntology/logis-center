@echo off
title GPU memory watch (close this window to stop)
:loop
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0gpu_mem_snapshot.ps1" watch
timeout /t 15 /nobreak > nul
goto loop
