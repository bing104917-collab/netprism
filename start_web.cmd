@echo off
rem ==========================================================================
rem  NetPrism Web mode - double-click this file.
rem  Starts the local server (examples\demo by default) and opens the browser.
rem  Close this window to stop the server.
rem
rem  Optional:  start_web.cmd <data-folder> --no-open
rem ==========================================================================
chcp 65001 >nul
setlocal
pushd "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "PY=%NETPRISM_PYTHON%"
if not defined PY set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=C:\Users\hrbkcy2523\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
if not exist "%PY%" set "PY=python"

"%PY%" -c "import sys" >nul 2>nul
if errorlevel 1 (
    echo.
    echo [NetPrism] 没有找到可用的 Python。
    echo             请安装 Python 3.10+，或设置 NETPRISM_PYTHON 指向 python.exe。
    echo.
    pause
    popd
    exit /b 1
)

"%PY%" "%~dp0scripts\start_web.py" %*
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" (
    echo.
    echo [NetPrism] 启动失败，退出代码 %EXITCODE%。
    pause
)
popd
exit /b %EXITCODE%
