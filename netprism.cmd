@echo off
rem NetPrism launcher for Windows.
rem Prefers NETPRISM_PYTHON, then the DSH bundled interpreter used in this
rem workspace, then whatever "python" is on PATH. Runs from the repo root so
rem the launcher works no matter which directory it is called from.
setlocal
pushd "%~dp0"
set "PY=%NETPRISM_PYTHON%"
if not defined PY set "PY=C:\Users\hrbkcy2523\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" -m netprism %*
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%
