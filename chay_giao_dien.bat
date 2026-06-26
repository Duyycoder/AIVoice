@echo off
setlocal

rem -- Xac dinh thu muc du an (luon lay thu muc chua file bat nay) --
set "PROJ_DIR=%~dp0"
rem Xoa dau \ o cuoi
if "%PROJ_DIR:~-1%"=="\" set "PROJ_DIR=%PROJ_DIR:~0,-1%"

cd /d "%PROJ_DIR%"

echo ============================================================
echo   AIVoice - Web UI
echo   Thu muc: %PROJ_DIR%
echo ============================================================
echo.

rem -- Kiem tra moi truong ao --
if not exist "%PROJ_DIR%\.venv\Scripts\python.exe" (
    echo [LOI] Khong tim thay .venv\Scripts\python.exe
    echo Vui long chay setup truoc.
    pause
    exit /b 1
)

echo [*] Dang khoi dong server...
echo [*] Trinh duyet se tu dong mo khi server san sang.
echo [*] Dung dong cua so nay khi dang su dung.
echo.

"%PROJ_DIR%\.venv\Scripts\python.exe" "%PROJ_DIR%\web_ui.py"

echo.
echo [!] Server da dung. Nhan phim bat ky de dong.
pause
