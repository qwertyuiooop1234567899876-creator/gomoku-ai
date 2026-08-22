@echo off
setlocal
cd /d "%~dp0"
python -m tools.build_native --clean
if errorlevel 1 (
  echo.
  echo NativeCore build failed.
  pause
  exit /b 1
)
python -m unittest tests.test_v0140_native_core -v
if errorlevel 1 (
  pause
  exit /b 1
)
echo.
echo NativeCore build and verification completed.
pause
