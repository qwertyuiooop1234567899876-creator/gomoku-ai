@echo off
setlocal
cd /d "%~dp0"
title YiXin Protocol Smoke Test V0.15.1

if not exist "yixin_smoke_test.py" goto no_program
if not exist "yixin\engine.exe" goto no_engine

python -c "import sys" >nul 2>nul
if %errorlevel%==0 set "PY_CMD=python"
if not defined PY_CMD py -3 -c "import sys" >nul 2>nul
if not defined PY_CMD if %errorlevel%==0 set "PY_CMD=py -3"
if not defined PY_CMD goto no_python

set "PYTHONUTF8=1"
%PY_CMD% -X utf8 yixin_smoke_test.py
set "EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %EXIT_CODE%

:no_program
echo ERROR: yixin_smoke_test.py was not found.
echo.
pause
exit /b 1

:no_engine
echo ERROR: yixin\engine.exe was not found.
echo.
pause
exit /b 1

:no_python
echo ERROR: Python was not found.
echo.
pause
exit /b 1
