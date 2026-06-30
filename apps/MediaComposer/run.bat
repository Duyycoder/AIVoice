@echo off
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

cd /d "%SCRIPT_DIR%"

set "PYTHONPATH=%SCRIPT_DIR%"
set "STREAMLIT_CMD=..\..\.venv\Scripts\python.exe -m streamlit"
echo Starting MediaComposer using parent virtual environment...
%STREAMLIT_CMD% run webui/Main.py --server.port 8502
pause

