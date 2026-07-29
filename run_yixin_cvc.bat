@echo off
setlocal
cd /d "%~dp0"
title Gomoku SearchAI vs YiXin V0.10.3

echo ========================================
echo      SearchAI vs YiXin Auto CVC
echo ========================================
echo.

if not exist "arena.py" goto no_program
if not exist "yixin\engine.exe" goto no_engine

where python >nul 2>nul
if %errorlevel%==0 goto use_python

where py >nul 2>nul
if %errorlevel%==0 goto use_py

goto no_python

:use_python
set "PY_CMD=python"
goto run_game

:use_py
set "PY_CMD=py -3"
goto run_game

:run_game
set "PYTHONUTF8=1"
%PY_CMD% -X utf8 arena.py --black search --white yixin --black-depth 8 --black-time-limit 60 --white-time-limit 10 --watch
set "EXIT_CODE=%ERRORLEVEL%"
echo.

if not "%EXIT_CODE%"=="0" echo CVC exited with error code %EXIT_CODE%.
if "%EXIT_CODE%"=="0" echo CVC finished normally.
echo.
pause
exit /b %EXIT_CODE%

:no_program
echo ERROR: arena.py was not found in:
echo %CD%
echo.
pause
exit /b 1

:no_engine
echo ERROR: yixin\engine.exe was not found.
echo Keep the supplied YiXin core in the yixin folder.
echo.
pause
exit /b 1

:no_python
echo ERROR: Python was not found.
echo Install Python or add it to PATH.
echo.
pause
exit /b 1
