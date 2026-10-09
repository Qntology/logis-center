@echo off
cd /d "%~dp0"
(
call "%~dp0_env_rocm.bat"
echo errorlevel=%errorlevel%
echo VCToolsVersion=%VCToolsVersion%
echo VCToolsInstallDir=%VCToolsInstallDir%
echo VCVARS_VER=%VCVARS_VER%
echo VS_SCRIPT=%VS_SCRIPT%
echo VS_ARGS=%VS_ARGS%
echo INCLUDE=%INCLUDE%
where cl
where link
) > diag_rocm.log 2>&1
