@echo off
chcp 65001 > nul
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "PYTHONIOENCODING=utf-8"

if not exist "%SystemRoot%\System32\vulkan-1.dll" (
    echo [Vulkan] vulkan-1.dll 이 없습니다. GPU 드라이버를 설치하면 함께 설치되며, 없으면 실행 시 CPU 로 동작합니다.
)

cd src-tauri

set "PATH=%CD%\dlls;%PATH%"

echo [DEV] Starting Tauri application (Vulkan)...
cargo tauri dev -- --no-default-features --features vulkan