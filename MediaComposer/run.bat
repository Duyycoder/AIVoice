@echo off
set "CURRENT_DIR=%CD%"
set "PYTHONPATH=%CURRENT_DIR%"
set "STREAMLIT_CMD=..\.venv\Scripts\python.exe -m streamlit"
echo Starting MediaComposer using parent virtual environment...
%STREAMLIT_CMD% run webui/Main.py --server.port 8502
pause
