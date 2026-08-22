@echo off
setlocal
cd /d "%~dp0"
title Gomoku AI - Board UI

if not exist "app\desktop_ui.py" goto no_program

if exist ".venv\Scripts\python.exe" set "PY_CMD=.venv\Scripts\python.exe"
if not defined PY_CMD python -c "import sys" >nul 2>nul
if not defined PY_CMD if %errorlevel%==0 set "PY_CMD=python"
if not defined PY_CMD py -3 -c "import sys" >nul 2>nul
if not defined PY_CMD if %errorlevel%==0 set "PY_CMD=py -3"
if not defined PY_CMD goto no_python

set "PYTHONUTF8=1"
%PY_CMD% -X utf8 -c "import tkinter as tk; root=tk.Tk(); root.withdraw(); root.destroy()" >nul 2>nul
if errorlevel 1 goto browser_ui

%PY_CMD% -X utf8 -m app.desktop_ui
goto ui_finished

:browser_ui
echo Tcl/Tk is unavailable. Starting the local browser UI.
%PY_CMD% -X utf8 -m app.web_ui

:ui_finished
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo UI exited with error code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%

:no_program
echo ERROR: app\desktop_ui.py was not found in:
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
