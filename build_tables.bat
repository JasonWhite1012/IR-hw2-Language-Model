@echo off
cd /d "%~dp0"
chcp 65001 > nul
py -3 "%~dp0build_tables.py"
set "exit_code=%errorlevel%"
echo.
pause
exit /b %exit_code%
