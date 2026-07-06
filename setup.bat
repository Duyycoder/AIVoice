@echo off
setlocal enabledelayedexpansion

rem -- Thiet lap thu muc Cache cuc bo trong du an --
set "PROJ_DIR=%~dp0"
if "%PROJ_DIR:~-1%"=="\" set "PROJ_DIR=%PROJ_DIR:~0,-1%"
set "HF_HOME=%PROJ_DIR%\models\.cache\huggingface"
set "TORCH_HOME=%PROJ_DIR%\models\.cache\torch"
set "XDG_CACHE_HOME=%PROJ_DIR%\models\.cache\xdg"

:: Thiet lap thu muc TEMP/TMP cuc bo sieu ngan de tranh loi Windows Long Path (vi du: llama-cpp-python)
set "TEMP_DIR=%SystemDrive%\.t"
mkdir "%TEMP_DIR%" >nul 2>&1
if not exist "%TEMP_DIR%" (
    set "TEMP_DIR=%USERPROFILE%\.t"
    mkdir "%TEMP_DIR%" >nul 2>&1
)
if not exist "%TEMP_DIR%" (
    set "TEMP_DIR=%PROJ_DIR%\.t"
    mkdir "%TEMP_DIR%" >nul 2>&1
)
set "TEMP=%TEMP_DIR%"
set "TMP=%TEMP_DIR%"

rem Mac dinh TAI model MediaComposer luon (idempotent - file da co thi bo qua)
rem de "pull ve + chay setup = dung duoc ngay". Dung --skip-models de bo qua.
set DOWNLOAD_MC_MODELS=1
if /i "%~1"=="--skip-models" set DOWNLOAD_MC_MODELS=0
if /i "%~2"=="--skip-models" set DOWNLOAD_MC_MODELS=0

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

set REUSE_SYSTEM=0
set RECREATE_VENV=0

:: Check if global python has PyTorch CUDA
set GLOBAL_TORCH_CUDA=0
%PYTHON_EXE% -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if !errorlevel! equ 0 (
    set GLOBAL_TORCH_CUDA=1
)

if exist .venv (
    rem .venv already exists, check if it has torch installed
    .venv\Scripts\python.exe -c "import torch" >nul 2>&1
    if !errorlevel! neq 0 (
        rem .venv exists but doesn't have torch, and global python has torch with CUDA
        if !GLOBAL_TORCH_CUDA! equ 1 (
            echo.
            echo [XAC NHAN] Thu muc .venv da ton tai nhung chua duoc cai dat PyTorch.
            echo Phat hien PyTorch ho tro CUDA da co san tren Python he thong.
            set /p CHOOSE_RECREATE="Ban co muon khoi tao lai .venv de ke thua PyTorch he thong (tranh phai tai lai 2.4 GB) khong? (Y/N) [Mac dinh: Y]: "
            if "!CHOOSE_RECREATE!"=="" set CHOOSE_RECREATE=Y
            if /i "!CHOOSE_RECREATE!"=="Y" (
                set RECREATE_VENV=1
                set REUSE_SYSTEM=1
            )
        )
    ) else (
        echo [INFO] Thu muc .venv da ton tai va da co PyTorch.
    )
) else (
    rem .venv does not exist
    if !GLOBAL_TORCH_CUDA! equ 1 (
        echo.
        echo [XAC NHAN] Phat hien PyTorch ho tro CUDA da co san tren Python he thong.
        set /p CHOOSE_REUSE="Ban co muon tao .venv ke thua PyTorch he thong (tranh phai tai lai 2.4 GB) khong? (Y/N) [Mac dinh: Y]: "
        if "!CHOOSE_REUSE!"=="" set CHOOSE_REUSE=Y
        if /i "!CHOOSE_REUSE!"=="Y" (
            set REUSE_SYSTEM=1
        )
    )
)

if !RECREATE_VENV! equ 1 (
    echo [INFO] Dang xoa thu muc .venv cu...
    rmdir /s /q .venv >nul 2>&1
)

if not exist .venv (
    if !REUSE_SYSTEM! equ 1 (
        echo [INFO] Khoi tao .venv ke thua thu vien he thong - system-site-packages...
        %PYTHON_EXE% -m venv --system-site-packages .venv
    ) else (
        echo [INFO] Khoi tao .venv sach hoan toan...
        %PYTHON_EXE% -m venv .venv
    )
    if !errorlevel! neq 0 (
        echo [ERROR] Khoi tao .venv that bai.
        pause
        exit /b 1
    )
    echo [INFO] Khoi tao .venv thanh cong.
)
echo.

:: 4. Install pip 24.0 (Required to bypass omegaconf metadata error)
echo ----------------------------------------------------------------------
echo [INFO] Dang thiet lap pip phien ban thich hop (pip 24.0)...
echo ----------------------------------------------------------------------
.venv\Scripts\python.exe -m pip install --default-timeout=1000 "pip==24.0"
if !errorlevel! neq 0 (
    echo [WARNING] Cai dat pip 24.0 that bai.
)
echo.



:: 5. Install libraries
echo ----------------------------------------------------------------------
echo [INFO] Dang kiem tra phan cung GPU de toi uu hoa cai dat...
echo ----------------------------------------------------------------------
rem Tu dong chon index PyTorch theo GPU:
rem   - RTX 50-Series (Blackwell sm_120): BAT BUOC cu128
rem   - GPU NVIDIA khac (RTX 20/30/40):   cu124
rem   - Khong co GPU NVIDIA:              bo qua, requirements.txt cai ban mac dinh
set TORCH_INDEX=
powershell -Command "if ((Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -match 'RTX 50') { exit 0 } else { exit 1 }" >nul 2>&1
if !errorlevel! equ 0 (
    set TORCH_INDEX=cu128
    echo [INFO] Phat hien GPU RTX 50-Series Blackwell - can PyTorch CUDA 12.8.
) else (
    powershell -Command "if ((Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -match 'NVIDIA') { exit 0 } else { exit 1 }" >nul 2>&1
    if !errorlevel! equ 0 (
        set TORCH_INDEX=cu124
        echo [INFO] Phat hien GPU NVIDIA - can PyTorch CUDA 12.4.
    ) else (
        echo [INFO] Khong phat hien GPU NVIDIA - dung PyTorch CPU mac dinh.
    )
)

if not "!TORCH_INDEX!"=="" (
    rem Kiem tra torch hien tai da dung ban CUDA chua; sai ban thi go va cai lai dong bo ca 3 goi
    .venv\Scripts\python.exe -c "import torch,sys; sys.exit(0 if '!TORCH_INDEX!' in torch.__version__ else 1)" >nul 2>&1
    if !errorlevel! neq 0 (
        echo [INFO] Cai dat PyTorch !TORCH_INDEX! - torch/torchvision/torchaudio dong bo...
        .venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio >nul 2>&1
        .venv\Scripts\python.exe -m pip install --default-timeout=1000 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/!TORCH_INDEX!
        if !errorlevel! neq 0 (
            echo [WARNING] Cai dat PyTorch !TORCH_INDEX! that bai. He thong se thu cai ban mac dinh tu requirements.
        ) else (
            echo [INFO] Cai dat PyTorch !TORCH_INDEX! thanh cong!
        )
    ) else (
        echo [INFO] PyTorch !TORCH_INDEX! da dung phien ban - bo qua buoc cai lai.
    )
)

rem RTX 50-Series: onnxruntime-gpu phai >= 1.22 moi co kernel Blackwell sm_120
rem (ban 1.20.x se khien InsightFace fallback ve CPU voi warning)
if "!TORCH_INDEX!"=="cu128" (
    echo [INFO] RTX 50-Series: nang cap onnxruntime-gpu ^>= 1.22 cho kernel Blackwell...
    .venv\Scripts\python.exe -m pip install --default-timeout=1000 "onnxruntime-gpu>=1.22"
    if !errorlevel! neq 0 (
        echo [WARNING] Nang cap onnxruntime-gpu that bai - InsightFace se chay CPU (khong anh huong engine CLIP mac dinh).
    )
)

echo.
echo ----------------------------------------------------------------------
echo [INFO] Dang cai dat cac thu vien Python (requirements.txt)...
echo ----------------------------------------------------------------------
.venv\Scripts\python.exe -m pip install --default-timeout=1000 -r requirements.txt --prefer-binary
if !errorlevel! neq 0 (
    echo [ERROR] Cai dat thu vien that bai.
    pause
    exit /b 1
)
echo [INFO] Cai dat thu vien thanh cong.
echo.

:: 5a2. Patch basicsr 1.4.2 tuong thich torchvision moi
::      (torchvision >=0.17 da xoa module functional_tensor -> basicsr import loi,
::       khien RealESRGAN am tham fallback ve PIL resize, giam chat luong upscale)
echo ----------------------------------------------------------------------
echo [INFO] Dang patch basicsr tuong thich torchvision moi...
echo ----------------------------------------------------------------------
.venv\Scripts\python.exe -c "import os,io; p=os.path.join('.venv','Lib','site-packages','basicsr','data','degradations.py'); s=io.open(p,encoding='utf-8').read(); n=s.replace('from torchvision.transforms.functional_tensor import rgb_to_grayscale','from torchvision.transforms.functional import rgb_to_grayscale'); io.open(p,'w',encoding='utf-8').write(n); print('[INFO] basicsr da duoc patch.' if n!=s else '[INFO] basicsr da patch san - bo qua.')"
if !errorlevel! neq 0 (
    echo [WARNING] Patch basicsr that bai - RealESRGAN co the khong hoat dong.
)
echo.

:: 5b. Install rvc-python separately with --no-deps
::     (rvc-python declares numpy<=1.23.5 which conflicts with coqui-tts's numpy>=1.26.0)
::     Its runtime dependencies are already installed via requirements.txt above.
echo ----------------------------------------------------------------------
echo [INFO] Dang cai dat rvc-python (--no-deps, tranh xung dot numpy)...
echo ----------------------------------------------------------------------
.venv\Scripts\python.exe -m pip install --default-timeout=1000 rvc-python>=0.1.5 --no-deps
if !errorlevel! neq 0 (
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
    .venv\Scripts\python.exe -m pip install --default-timeout=1000 git+https://github.com/vunb/viphoneme.git git+https://github.com/vunb/vinorm.git
    if !errorlevel! neq 0 (
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
    if not exist "third_party\Kokoro-Vietnamese" (
        git clone https://github.com/iamdinhthuan/Kokoro-Vietnamese.git third_party\Kokoro-Vietnamese
    )
    if exist "third_party\Kokoro-Vietnamese" (
        cd third_party\Kokoro-Vietnamese
        ..\..\.venv\Scripts\pip install --default-timeout=1000 -e .
        if !errorlevel! neq 0 (
            echo [WARNING] Cấu hình Kokoro-Vietnamese thất bại.
        ) else (
            echo [INFO] Cai dat Kokoro-Vietnamese thanh cong.
        )
        cd ..\..
    ) else (
        echo [WARNING] Khong the clone Kokoro-Vietnamese tu GitHub.
    )
    echo.
)

:: 6d. Install Valtec-TTS (requires Git)
if !GIT_OK! equ 1 (
    echo ----------------------------------------------------------------------
    echo [INFO] Dang thiet lap va cai dat Valtec-TTS...
    echo ----------------------------------------------------------------------
    if not exist "third_party\valtec-tts" (
        git clone https://github.com/tronghieuit/valtec-tts.git third_party\valtec-tts
    )
    if exist "third_party\valtec-tts" (
        cd third_party\valtec-tts
        ..\..\.venv\Scripts\pip install --default-timeout=1000 -e .
        if !errorlevel! neq 0 (
            echo [WARNING] Cấu hình Valtec-TTS thất bại.
        ) else (
            echo [INFO] Cai dat Valtec-TTS thanh cong.
        )
        cd ..\..
    ) else (
        echo [WARNING] Khong the clone Valtec-TTS tu GitHub.
    )
    echo.
)


:: 6b. Copy config.toml from example if not exist
if not exist "apps\MediaComposer\config.toml" (
    echo ----------------------------------------------------------------------
    echo [INFO] Dang khoi tao config.toml tu file mau...
    echo ----------------------------------------------------------------------
    copy "apps\MediaComposer\config.toml.example" "apps\MediaComposer\config.toml" >nul
    if !errorlevel! equ 0 (
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
.venv\Scripts\python.exe src\download_models.py --engine all
if %errorlevel% neq 0 (
    echo [WARNING] Qua trinh tai mo hinh bi gian doan.
)
echo.

:: 7b. Check/Download MediaComposer Storytelling Models
echo ----------------------------------------------------------------------
echo [INFO] Kiem tra va tai mo hinh AI Storytelling (RealESRGAN, IP-Adapter)...
echo ----------------------------------------------------------------------
if !DOWNLOAD_MC_MODELS! equ 1 (
    .venv\Scripts\python.exe apps\MediaComposer\app\services\model_downloader.py --download
    if !errorlevel! neq 0 (
        echo [WARNING] Mot so model tai that bai - ung dung se tu dong tai lai khi can (can mang).
    )
) else (
    .venv\Scripts\python.exe apps\MediaComposer\app\services\model_downloader.py --check-only
    if !errorlevel! neq 0 (
        echo [WARNING] Mot so model Storytelling chua co - ung dung se tu dong tai khi can.
    )
)
echo.

:: 8. Hardware & GPU Diagnostic
echo ----------------------------------------------------------------------
echo [INFO] Dang tien hanh chan doan GPU CUDA...
echo ----------------------------------------------------------------------
.venv\Scripts\python.exe src\check_gpu.py
echo.

echo ======================================================================
echo THIET LAP THANH CONG! AIVoice da san sang hoat dong.
echo Chay ung dung bang lenh:
echo.
echo     .venv\Scripts\python.exe src\main.py
echo.
echo ======================================================================
pause
