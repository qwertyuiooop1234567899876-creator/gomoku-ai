@echo off
setlocal
cd /d "%~dp0"
title Gomoku AI V0.15.0

if not exist "main.py" goto no_program

if exist ".venv\Scripts\python.exe" set "PY_CMD=.venv\Scripts\python.exe"
if not defined PY_CMD python -c "import sys" >nul 2>nul
if not defined PY_CMD if %errorlevel%==0 set "PY_CMD=python"
if not defined PY_CMD py -3 -c "import sys" >nul 2>nul
if not defined PY_CMD if %errorlevel%==0 set "PY_CMD=py -3"
if not defined PY_CMD goto no_python

set "PYTHONUTF8=1"
%PY_CMD% -X utf8 main.py
set "EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %EXIT_CODE%

:no_program
echo ERROR: main.py was not found in:
echo %CD%
echo.
pause
exit /b 1

:no_python
echo ERROR: no executable Python interpreter was found.
echo Install Python, add it to PATH, or create .venv.
echo.
pause
exit /b 1
