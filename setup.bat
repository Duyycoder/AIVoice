@echo off
setlocal enabledelayedexpansion

echo ======================================================================
echo          AIVoice Auto-Setup Tool for Windows (Python 3.11)
echo ======================================================================
echo.

:: 1. Check Python installation
set PYTHON_EXE=
python --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%i in ('python --version') do (
        set py_ver=%%i
    )
    echo [INFO] Phat hien Python !py_ver! trong PATH.
    if "!py_ver:~0,4!"=="3.11" (
        set PYTHON_EXE=python
        goto :python_ok
    ) else (
        echo [WARNING] AIVoice yeu cau Python 3.11.
    )
)

:: Check via Python Launcher (py -3.11)
py -3.11 -c "import sys" >nul 2>&1
if %errorlevel% equ 0 (
    for /f "delims=" %%i in ('py -3.11 -c "import sys; print(sys.executable)"') do (
        set PYTHON_EXE="%%i"
    )
    echo [INFO] Tim thay Python 3.11 thong qua Python Launcher tai: !PYTHON_EXE!
    goto :python_ok
)

:: Check if installed in default Local AppData folder for User
set LOCAL_PY_EXE="%LocalAppData%\Programs\Python\Python311\python.exe"
if exist %LOCAL_PY_EXE% (
    set PYTHON_EXE=%LOCAL_PY_EXE%
    echo [INFO] Tim thay Python 3.11 tai: %LOCAL_PY_EXE%
    goto :python_ok
)

:: Check in default Program Files (for all users installation)
set SYSTEM_PY_EXE="%ProgramFiles%\Python311\python.exe"
if exist %SYSTEM_PY_EXE% (
    set PYTHON_EXE=%SYSTEM_PY_EXE%
    echo [INFO] Tim thay Python 3.11 tai: %SYSTEM_PY_EXE%
    goto :python_ok
)

:: If Python 3.11 is not found, download and install it silently
echo.
echo ----------------------------------------------------------------------
echo [INFO] Khong tim thay Python 3.11 tren may tinh nay.
echo Dang tu dong tai xuong Python 3.11.9 tu python.org...
echo ----------------------------------------------------------------------
echo.

:: Download Python 3.11.9 Installer
curl -L -o python-3.11.9-amd64.exe https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
if %errorlevel% neq 0 (
    echo [ERROR] Tai xuong Python that bai. Vui long kiem tra ket noi mang.
    pause
    exit /b 1
)

echo [INFO] Dang cai dat Python 3.11.9 chay ngam (Silent Mode)...
echo Vui long cho 1-2 phut...
start /wait python-3.11.9-amd64.exe /quiet PrependPath=1 Include_test=0
del python-3.11.9-amd64.exe

:: Verify silent install
if exist %LOCAL_PY_EXE% (
    set PYTHON_EXE=%LOCAL_PY_EXE%
    echo [INFO] Cai dat Python 3.11.9 thanh cong!
    goto :python_ok
) else if exist %SYSTEM_PY_EXE% (
    set PYTHON_EXE=%SYSTEM_PY_EXE%
    echo [INFO] Cai dat Python 3.11.9 thanh cong!
    goto :python_ok
) else (
    echo [ERROR] Cai dat Python tu dong that bai.
    echo Vui long tai va cai dat Python 3.11.9 thu cong tu: https://www.python.org/downloads/
    echo Nho tich chon "Add Python to PATH" khi cai dat.
    pause
    exit /b 1
)

:python_ok
echo [INFO] Su dung Python: %PYTHON_EXE%
echo.

:: 2. Check Git installation
set GIT_OK=0
git --version >nul 2>&1
if %errorlevel% equ 0 (
    set GIT_OK=1
    echo [INFO] Phat hien Git da duoc cai dat.
) else (
    echo [WARNING] Khong tim thay Git. Se bo qua cai dat phien am tu Github.
)
echo.

:: 3. Create virtual environment (.venv)
echo ----------------------------------------------------------------------
echo [INFO] Dang khoi tao moi truong ao (.venv)...
echo ----------------------------------------------------------------------
if not exist .venv (
    %PYTHON_EXE% -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Khoi tao .venv that bai.
        pause
        exit /b 1
    )
    echo [INFO] Khoi tao .venv thanh cong.
) else (
    echo [INFO] Thu muc .venv da ton tai.
)
echo.

:: 4. Install pip 24.0 (Required to bypass omegaconf metadata error)
echo ----------------------------------------------------------------------
echo [INFO] Dang thiet lap pip phien ban thich hop (pip 24.0)...
echo ----------------------------------------------------------------------
.venv\Scripts\python.exe -m pip install "pip==24.0"
if %errorlevel% neq 0 (
    echo [WARNING] Cai dat pip 24.0 that bai.
)
echo.

:: 5. Install libraries
echo ----------------------------------------------------------------------
echo [INFO] Dang cai dat cac thu vien Python (requirements.txt)...
echo ----------------------------------------------------------------------
.venv\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Cai dat thu vien that bai.
    pause
    exit /b 1
)
echo [INFO] Cai dat thu vien thanh cong.
echo.

:: 5b. Install rvc-python separately with --no-deps
::     (rvc-python declares numpy<=1.23.5 which conflicts with coqui-tts's numpy>=1.26.0)
::     Its runtime dependencies are already installed via requirements.txt above.
echo ----------------------------------------------------------------------
echo [INFO] Dang cai dat rvc-python (--no-deps, tranh xung dot numpy)...
echo ----------------------------------------------------------------------
.venv\Scripts\python.exe -m pip install rvc-python>=0.1.5 --no-deps
if %errorlevel% neq 0 (
    echo [WARNING] Cai dat rvc-python that bai.
) else (
    echo [INFO] Cai dat rvc-python thanh cong.
)
echo.

:: 6. Install optional Vietnamese phoneme packages (requires Git)
if !GIT_OK! equ 1 (
    echo ----------------------------------------------------------------------
    echo [INFO] Dang cai dat thu vien phien am tieng Viet viphoneme...
    echo ----------------------------------------------------------------------
    .venv\Scripts\python.exe -m pip install git+https://github.com/vunb/viphoneme.git git+https://github.com/vunb/vinorm.git
    if %errorlevel% neq 0 (
        echo [WARNING] Cai dat thu vien phien am tu GitHub that bai.
    ) else (
        echo [INFO] Cai dat thu vien phien am thanh cong.
    )
    echo.
)

:: 6c. Install Kokoro-Vietnamese (requires Git)
if !GIT_OK! equ 1 (
    echo ----------------------------------------------------------------------
    echo [INFO] Dang thiet lap va cai dat Kokoro-Vietnamese...
    echo ----------------------------------------------------------------------
    if not exist "Kokoro-Vietnamese" (
        git clone https://github.com/iamdinhthuan/Kokoro-Vietnamese.git
    )
    if exist "Kokoro-Vietnamese" (
        cd Kokoro-Vietnamese
        ..\.venv\Scripts\pip install -e .
        cd ..
        echo [INFO] Cai dat Kokoro-Vietnamese thanh cong.
    ) else (
        echo [WARNING] Khong the clone Kokoro-Vietnamese tu GitHub.
    )
    echo.
)

:: 6b. Copy config.toml from example if not exist
if not exist "MediaComposer\config.toml" (
    echo ----------------------------------------------------------------------
    echo [INFO] Dang khoi tao config.toml tu file mau...
    echo ----------------------------------------------------------------------
    copy "MediaComposer\config.toml.example" "MediaComposer\config.toml" >nul
    if %errorlevel% equ 0 (
        echo [INFO] Khoi tao config.toml thanh cong! Vui long cap nhat API key neu can.
    ) else (
        echo [WARNING] Khong the tu dong sao chep config.toml.
    )
    echo.
)

:: 7. Download model weights
echo ----------------------------------------------------------------------
echo [INFO] Dang tai trong so mo hinh AI (Piper & XTTSv2)...
echo ----------------------------------------------------------------------
.venv\Scripts\python.exe download_models.py --engine all
if %errorlevel% neq 0 (
    echo [WARNING] Qua trinh tai mo hinh bi gian doan.
)
echo.

:: 8. Hardware & GPU Diagnostic
echo ----------------------------------------------------------------------
echo [INFO] Dang tien hanh chan doan GPU CUDA...
echo ----------------------------------------------------------------------
.venv\Scripts\python.exe check_gpu.py
echo.

echo ======================================================================
echo THIET LAP THANH CONG! AIVoice da san sang hoat dong.
echo Chay ung dung bang lenh:
echo.
echo     .venv\Scripts\python.exe main.py
echo.
echo ======================================================================
pause
