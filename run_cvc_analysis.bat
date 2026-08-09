@echo off
setlocal
cd /d "%~dp0"
title Gomoku CVC YiXin Analysis V0.15.1

echo ========================================
echo         CVC YiXin Analysis
echo ========================================
echo.

if not exist "cvc_analysis.py" goto no_program
if not exist "yixin\engine.exe" goto no_engine

set "RECORD_PATH=%~1"
if not defined RECORD_PATH set /p "RECORD_PATH=Drag a CVC JSON here or enter its path: "
if not defined RECORD_PATH goto no_record
if not exist "%RECORD_PATH%" goto bad_record

python -c "import sys" >nul 2>nul
if %errorlevel%==0 goto use_python

py -3 -c "import sys" >nul 2>nul
if %errorlevel%==0 goto use_py

goto no_python

:use_python
set "PY_CMD=python"
goto run_analysis

:use_py
set "PY_CMD=py -3"
goto run_analysis

:run_analysis
set "PYTHONUTF8=1"
%PY_CMD% -X utf8 cvc_analysis.py "%RECORD_PATH%"
set "EXIT_CODE=%ERRORLEVEL%"
if "%EXIT_CODE%"=="0" exit /b 0

echo.
echo Analysis exited with error code %EXIT_CODE%.
echo.
pause
exit /b %EXIT_CODE%

:no_program
echo ERROR: cvc_analysis.py was not found.
echo.
pause
exit /b 1

:no_engine
echo ERROR: yixin\engine.exe was not found.
echo.
pause
exit /b 1

:no_record
echo ERROR: No JSON record was selected.
echo.
pause
exit /b 1

:bad_record
echo ERROR: Record not found:
echo %RECORD_PATH%
echo.
pause
exit /b 1

:no_python
echo ERROR: Python was not found.
echo.
pause
exit /b 1
