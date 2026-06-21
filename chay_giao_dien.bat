@echo off
setlocal enabledelayedexpansion

pushd "%~dp0"
set "PROJ_DIR=%cd%"
popd

if not exist "%PROJ_DIR%\.venv\Scripts\python.exe" (
    set "PROJ_DIR=F:\programfiles\AIVoice"
)

echo ============================================================
echo  AIVoice - Starting Web UI
echo  Project Directory: !PROJ_DIR!
echo ============================================================

if not exist "!PROJ_DIR!\.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at: !PROJ_DIR!\.venv
    echo Please check your project path.
    pause
    exit /b 1
)

cd /d "!PROJ_DIR!"

rem Start browser in background after 2 seconds delay
start /b cmd /c "timeout /t 2 > nul && start http://127.0.0.1:5000"

rem Start Flask server
.venv\Scripts\python.exe web_ui.py
pause
