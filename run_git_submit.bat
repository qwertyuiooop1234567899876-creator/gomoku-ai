@echo off
setlocal
cd /d "%~dp0"

python -m tools.git_submit
