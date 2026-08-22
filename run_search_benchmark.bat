@echo off
setlocal
cd /d "%~dp0"

python -m tools.search_benchmark --repeat 3 --json search-benchmark-results.json
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo SearchAI benchmark failed with error code %EXIT_CODE%.
) else (
    echo SearchAI benchmark: OK
)
pause
exit /b %EXIT_CODE%
