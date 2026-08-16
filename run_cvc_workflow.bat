@echo off
setlocal
cd /d "%~dp0"
title Gomoku Full CVC Workflow V0.16.3

echo ========================================
echo       Gomoku Full CVC Workflow
echo ========================================
echo.
echo 1. SearchAI vs SearchAI
echo 2. Analyze the new self-play record
echo 3. SearchAI vs YiXin
echo 4. Analyze the new YiXin record
echo.

if not exist "cvc_workflow.py" goto no_program
if not exist "cvc_analysis.py" goto no_program
if not exist "yixin\engine.exe" goto no_engine

python -c "import sys" >nul 2>nul
if %errorlevel%==0 goto use_python

py -3 -c "import sys" >nul 2>nul
if %errorlevel%==0 goto use_py

goto no_python

:use_python
set "PY_CMD=python"
goto run_workflow

:use_py
set "PY_CMD=py -3"
goto run_workflow

:run_workflow
set "PYTHONUTF8=1"
%PY_CMD% -X utf8 cvc_workflow.py
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" exit /b 0

echo.
echo Workflow exited with error code %EXIT_CODE%.
echo The window will remain open for inspection.
echo.
pause
exit /b %EXIT_CODE%

:no_program
echo ERROR: cvc_workflow.py or cvc_analysis.py was not found in:
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
