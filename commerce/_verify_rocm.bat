@echo off
setlocal EnableExtensions
chcp 65001 > nul
rem =====================================================================
rem  AMD ROCm verification runner. Every step writes a log that Claude reads.
rem  Usage: _verify_rocm.bat [env] [build] [fork] [test] [dev]
rem         (no arguments = all steps in this order)
rem  Logs : src-tauri\target\verify_rocm\
rem =====================================================================
set "ROOT=%~dp0"
set "LOGDIR=%ROOT%src-tauri\target\verify_rocm"
set "FORK=%ROOT%..\..\candle-rocm-vulkan"
set "FEATS=rocm,flash-attn,turbo-quant"
set "RUST_BACKTRACE=1"
set "STAGES=%*"
if "%STAGES%"=="" set "STAGES=env build fork test dev"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
set "SUMMARY=%LOGDIR%\summary.log"
echo [VERIFY] %date% %time% stages: %STAGES% > "%SUMMARY%"

echo [VERIFY] Setting up ROCm / MSVC environment...
call "%ROOT%_env_rocm.bat" > "%LOGDIR%\00_env_setup.log" 2>&1
if errorlevel 1 goto :env_failed
echo [VERIFY] ROCm: %ROCM_PATH% (%ROCM_TRACK%)
echo env_setup OK  ROCM_PATH=%ROCM_PATH% TRACK=%ROCM_TRACK% >> "%SUMMARY%"
set "PATH=%ROOT%src-tauri\dlls;%PATH%"

for %%S in (%STAGES%) do call :stage_%%S
echo.
echo ===================== SUMMARY =====================
type "%SUMMARY%"
echo ===================================================
echo Logs: %LOGDIR%
echo Tell Claude it is done.
pause
exit /b 0

:env_failed
echo env_setup FAIL (see 00_env_setup.log) >> "%SUMMARY%"
type "%LOGDIR%\00_env_setup.log"
echo [VERIFY] ROCm environment setup failed. Tell Claude it is done.
pause
exit /b 1

rem ---------------------------------------------------------------------
:stage_env
echo [VERIFY] %time% env: GPU / toolchain info...
set "LOG=%LOGDIR%\01_env.log"
echo === ROCM_PATH=%ROCM_PATH% ROCM_VER=%ROCM_VER% TRACK=%ROCM_TRACK% HIP_PATH=%HIP_PATH% > "%LOG%"
echo === HIP_ARCH=%HIP_ARCH% CANDLE_ROCM_TRACK=%CANDLE_ROCM_TRACK% >> "%LOG%"
where hipcc hipInfo amdgpu-arch >> "%LOG%" 2>&1
rustc -V >> "%LOG%" 2>&1
cargo -V >> "%LOG%" 2>&1
echo === hipInfo >> "%LOG%"
hipInfo >> "%LOG%" 2>&1
echo === amdgpu-arch >> "%LOG%"
amdgpu-arch >> "%LOG%" 2>&1
echo === Win32_VideoController >> "%LOG%"
powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Format-List Name,DriverVersion,AdapterRAM,VideoProcessor" >> "%LOG%" 2>&1
echo env DONE >> "%SUMMARY%"
exit /b 0

rem ---------------------------------------------------------------------
:stage_build
echo [VERIFY] %time% build: cargo build --features %FEATS% (first build takes a while)...
cd /d "%ROOT%src-tauri"
cargo build --no-default-features --features %FEATS% > "%LOGDIR%\02_app_build.log" 2>&1
set "RC=%errorlevel%"
call :record build %RC% 02_app_build.log
if not "%RC%"=="0" set "BUILD_FAILED=1"
exit /b 0

rem ---------------------------------------------------------------------
:stage_fork
echo [VERIFY] %time% fork: candle-core / candle-nn ROCm tests...
pushd "%FORK%"
cargo test -p candle-core --features rocm --no-fail-fast --test rocm_tests --test delta_rule_tests --test quantized_tests --test matmul_tests --test tensor_tests > "%LOGDIR%\03_fork_core_tests.log" 2>&1
set "RC=%errorlevel%"
call :record fork_core_tests %RC% 03_fork_core_tests.log
cargo test -p candle-nn --features rocm --no-fail-fast --test rocm_ops > "%LOGDIR%\04_fork_nn_tests.log" 2>&1
set "RC=%errorlevel%"
call :record fork_nn_tests %RC% 04_fork_nn_tests.log
popd
exit /b 0

rem ---------------------------------------------------------------------
:stage_test
if defined BUILD_FAILED (
    echo test SKIP build failed >> "%SUMMARY%"
    exit /b 0
)
echo [VERIFY] %time% test: app GPU tests (app_embed, mem_lifecycle)...
cd /d "%ROOT%src-tauri"
set "LOGIS_TEST_MODELS=1"
cargo test --test it --no-default-features --features %FEATS% --no-fail-fast -- app_embed mem_lifecycle --include-ignored --nocapture --test-threads=1 > "%LOGDIR%\05_app_tests.log" 2>&1
set "RC=%errorlevel%"
set "LOGIS_TEST_MODELS="
call :record app_tests %RC% 05_app_tests.log
exit /b 0

rem ---------------------------------------------------------------------
:stage_dev
if defined BUILD_FAILED (
    echo dev SKIP build failed >> "%SUMMARY%"
    exit /b 0
)
echo [VERIFY] %time% dev: starting the app with ROCm.
echo          Load the models and run a search / embedding once, then close the app window.
cd /d "%ROOT%src-tauri"
call %TAURI_CLI% dev -- --no-default-features --features %FEATS% > "%LOGDIR%\06_app_dev.log" 2>&1
set "RC=%errorlevel%"
call :record dev %RC% 06_app_dev.log
exit /b 0

rem ---------------------------------------------------------------------
:record
if "%~2"=="0" (
    echo %~1 OK  [%~3] >> "%SUMMARY%"
    echo [VERIFY] %time% %~1 OK
) else (
    echo %~1 FAIL rc=%~2 [%~3] >> "%SUMMARY%"
    echo [VERIFY] %time% %~1 FAIL rc=%~2  see %~3
)
exit /b 0
