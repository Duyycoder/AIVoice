@echo off
setlocal enabledelayedexpansion

pushd "%~dp0"
set "PROJ_DIR=%cd%"
popd

if not exist "%PROJ_DIR%\.venv\Scripts\python.exe" (
    set "PROJ_DIR=F:\programfiles\AIVoice"
)

echo ============================================================
echo  AIVoice - Running Integration Tests Suite
echo  Project Directory: !PROJ_DIR!
echo ============================================================

if not exist "!PROJ_DIR!\.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at: !PROJ_DIR!\.venv
    pause
    exit /b 1
)

cd /d "!PROJ_DIR!"
.venv\Scripts\python.exe tests/run_tests.py
pause
