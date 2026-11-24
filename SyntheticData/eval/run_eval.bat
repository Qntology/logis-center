@echo off
rem =====================================================================
rem  Synthetic data evaluation runner (CUDA / ROCm).
rem  Set LOGIS_EVAL_BACKEND=cuda or rocm (default: rocm, or auto-cuda if CUDA_PATH set and ROCm missing).
rem  Set EVAL_PROFILE=release or debug (default: release).
rem  Build first with:
rem    CUDA: scripts\test_cargo.bat --release --no-run   (or debug: scripts\test_cargo.bat --no-run)
rem    ROCm: scripts\test_cargo_rocm.bat --no-run
rem
rem  Usage:
rem    run_eval.bat <name> plan <plan.jsonl> <result.jsonl> <log>
rem    run_eval.bat <name> vision <png dir> <result.jsonl> <log> [anchor language] [dump dir]
rem    run_eval.bat <name> embed <texts.json> <out.f32> <log> <granite or siglip2>
rem    run_eval.bat <name> gates <cases dir> <result.jsonl> <log> [block or line]
rem    run_eval.bat <name> names <cases dir> <result.jsonl> <log>
rem    run_eval.bat <name> labels <cases dir> <result.jsonl> <log> [cpu]
rem      (cases dir = make_vision_cases.py output; gates needs no model weights, labels loads granite)
rem
rem  Set EVAL_EXE to a copied test binary to keep a baseline runnable while the code changes.
rem  Each run lives in %LOCALAPPDATA%\logis-center-eval\<name>. Only "models" is a junction to the
rem  app's model folder. DB and SDS ledgers start empty there, so the app's data is never touched.
rem =====================================================================
setlocal
chcp 65001 > nul
if "%~1"=="" (
    echo Usage: %~nx0 ^<name^> ^<plan^|census^|seedids^|embed^|gates^|names^|labels^|vision^> ...
    exit /b 1
)
set "ROOT=%~dp0..\.."
set "BACKEND=%LOGIS_EVAL_BACKEND%"
if not defined BACKEND (
    if exist "C:\Program Files\AMD\ROCm" (
        set "BACKEND=rocm"
    ) else (
        set "BACKEND=cuda"
    )
)
if /i "%BACKEND%"=="cuda" (
    call "%ROOT%\scripts\_env_cuda.bat" > nul
    if errorlevel 1 exit /b 1
) else if /i "%BACKEND%"=="vulkan" (
    call "%ROOT%\scripts\_env_vulkan.bat" > nul
    if errorlevel 1 exit /b 1
) else (
    call "%ROOT%\scripts\_env_rocm.bat" > nul
    if errorlevel 1 exit /b 1
)
set "PATH=%ROOT%\apps\desktop\dlls;%PATH%"
set "RUST_BACKTRACE=1"

set "EVAL_ROOT=%LOCALAPPDATA%\logis-center-eval\%~1"
if not exist "%EVAL_ROOT%" mkdir "%EVAL_ROOT%"
if not exist "%EVAL_ROOT%\models" mklink /J "%EVAL_ROOT%\models" "%LOCALAPPDATA%\logis-center\models" > nul
set "LOGIS_APP_DIR=%EVAL_ROOT%"

set "PROFILE=%EVAL_PROFILE%"
if not defined PROFILE set "PROFILE=release"
set "EXE=%EVAL_EXE%"
for /f "delims=" %%F in ('dir /b /o-d "%ROOT%\target\%PROFILE%\deps\it-*.exe" 2^>nul') do if not defined EXE set "EXE=%ROOT%\target\%PROFILE%\deps\%%F"
if not defined EXE (
    for /f "delims=" %%F in ('dir /b /o-d "%ROOT%\target\release\deps\it-*.exe" 2^>nul') do if not defined EXE set "EXE=%ROOT%\target\release\deps\%%F"
)
if not defined EXE (
    for /f "delims=" %%F in ('dir /b /o-d "%ROOT%\target\debug\deps\it-*.exe" 2^>nul') do if not defined EXE set "EXE=%ROOT%\target\debug\deps\%%F"
)
if not defined EXE (
    echo No test binary. Run scripts\test_cargo.bat --no-run ^(CUDA^) or scripts\test_cargo_rocm.bat --no-run ^(ROCm^) first.
    exit /b 1
)

cd /d "%ROOT%\apps\desktop"
if /i "%~2"=="plan" (
    set "LOGIS_SYNTH_PLAN=%~3"
    set "LOGIS_SYNTH_OUT=%~4"
    "%EXE%" synthetic_eval::run_plan --ignored --nocapture --test-threads 1 > "%~5" 2>&1
) else if /i "%~2"=="census" (
    set "LOGIS_SYNTH_PLAN=%~3"
    set "LOGIS_SYNTH_OUT=%~4"
    "%EXE%" synthetic_eval::census_probe --ignored --nocapture --test-threads 1 > "%~5" 2>&1
) else if /i "%~2"=="seedids" (
    set "LOGIS_SYNTH_PLAN=%~3"
    set "LOGIS_SYNTH_OUT=%~4"
    "%EXE%" synthetic_eval::seed_ids --ignored --nocapture --test-threads 1 > "%~5" 2>&1
) else if /i "%~2"=="embed" (
    set "LOGIS_SYNTH_TEXTS=%~3"
    set "LOGIS_SYNTH_OUT=%~4"
    set "LOGIS_SYNTH_MODEL=%~6"
    "%EXE%" synthetic_eval::embed_dump --ignored --nocapture --test-threads 1 > "%~5" 2>&1
) else if /i "%~2"=="gates" (
    set "LOGIS_SYNTH_CASES=%~3"
    set "LOGIS_SYNTH_OUT=%~4"
    set "LOGIS_SYNTH_BBOX=%~6"
    "%EXE%" synthetic_eval::vision_gates --ignored --nocapture --test-threads 1 > "%~5" 2>&1
) else if /i "%~2"=="names" (
    set "LOGIS_SYNTH_CASES=%~3"
    set "LOGIS_SYNTH_OUT=%~4"
    "%EXE%" synthetic_eval::name_residue_cases --ignored --nocapture --test-threads 1 > "%~5" 2>&1
) else if /i "%~2"=="labels" (
    set "LOGIS_SYNTH_CASES=%~3"
    set "LOGIS_SYNTH_OUT=%~4"
    if /i "%~6"=="cpu" set "LOGIS_SYNTH_CPU=1"
    "%EXE%" synthetic_eval::label_gates --ignored --nocapture --test-threads 1 > "%~5" 2>&1
) else (
    set "LOGIS_SYNTH_PAGES=%~3"
    set "LOGIS_SYNTH_OUT=%~4"
    set "LOGIS_SYNTH_LANG=%~6"
    set "LOGIS_SYNTH_DUMP=%~7"
    "%EXE%" synthetic_eval::vision_probe --ignored --nocapture --test-threads 1 > "%~5" 2>&1
)
exit /b %errorlevel%
