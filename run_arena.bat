@echo off
setlocal
cd /d "%~dp0"
title Gomoku AI Arena V0.11.0

echo ========================================
echo        Gomoku AI Arena V0.11.0
echo ========================================
echo.

if not exist "arena.py" goto no_arena

where python >nul 2>nul
if %errorlevel%==0 goto use_python

where py >nul 2>nul
if %errorlevel%==0 goto use_py

goto no_python

:use_python
set "PY_CMD=python"
goto run_arena

:use_py
set "PY_CMD=py -3"
goto run_arena

:run_arena
set "PYTHONUTF8=1"
%PY_CMD% -X utf8 arena.py
set "EXIT_CODE=%ERRORLEVEL%"
echo.

if not "%EXIT_CODE%"=="0" echo Arena exited with error code %EXIT_CODE%.
if "%EXIT_CODE%"=="0" echo Arena finished normally.

echo.
pause
exit /b %EXIT_CODE%

:no_arena
echo ERROR: arena.py was not found in:
echo %CD%
echo.
echo Put this BAT file in the gomoku-ai project folder.
echo.
pause
exit /b 1

:no_python
echo ERROR: Python was not found.
echo Install Python or add it to PATH.
echo.
pause
exit /b 1
