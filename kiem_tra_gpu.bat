@echo off
setlocal enabledelayedexpansion

pushd "%~dp0"
set "PROJ_DIR=%cd%"
popd

echo ============================================================
echo  AIVoice - Checking GPU and NVIDIA CUDA status
echo  Project Directory: !PROJ_DIR!
echo ============================================================

if not exist "!PROJ_DIR!\.venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at: !PROJ_DIR!\.venv
    echo Vui long chay setup.bat truoc de cai dat moi truong ao.
    pause
    exit /b 1
)

cd /d "!PROJ_DIR!"
.venv\Scripts\python.exe check_gpu.py
pause
