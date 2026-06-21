@echo off
:: Set console to UTF-8 to display Vietnamese characters correctly
chcp 65001 > nul
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
    echo Phát hiện Python version !py_ver! trong hệ thống (PATH).
    if "!py_ver:~0,4!"=="3.11" (
        set PYTHON_EXE=python
        goto :python_ok
    ) else (
        echo Cảnh báo: AIVoice yêu cầu Python 3.11 để tương thích với các thư viện đã build sẵn.
        echo Phiên bản hiện tại (!py_ver!) có thể gây lỗi.
    )
)

:: Check if installed in default Local AppData folder for User
set LOCAL_PY_EXE="%LocalAppData%\Programs\Python\Python311\python.exe"
if exist %LOCAL_PY_EXE% (
    set PYTHON_EXE=%LOCAL_PY_EXE%
    echo Tìm thấy Python 3.11 được cài đặt tại: %LOCAL_PY_EXE%
    goto :python_ok
)

:: If Python 3.11 is not found, download and install it silently
echo.
echo ----------------------------------------------------------------------
echo Không tìm thấy Python 3.11 trên máy tính này.
echo Hệ thống sẽ tự động tải xuống Python 3.11.9 từ python.org...
echo ----------------------------------------------------------------------
echo.

:: Download Python 3.11.9 Installer
curl -L -o python-3.11.9-amd64.exe https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
if %errorlevel% neq 0 (
    echo [LỖI] Tải xuống Python thất bại. Vui lòng kiểm tra lại kết nối internet.
    pause
    exit /b 1
)

echo Đang cài đặt Python 3.11.9 ở chế độ chạy ngầm (Silent Mode)...
echo Quá trình cài đặt diễn ra trong khoảng 1-2 phút, vui lòng đợi...
start /wait python-3.11.9-amd64.exe /quiet PrependPath=1 Include_test=0
del python-3.11.9-amd64.exe

:: Verify silent install
if exist %LOCAL_PY_EXE% (
    set PYTHON_EXE=%LOCAL_PY_EXE%
    echo Cài đặt Python 3.11.9 thành công!
    goto :python_ok
) else (
    echo [LỖI] Cài đặt Python tự động thất bại hoặc được cài ở thư mục không mặc định.
    echo Vui lòng tải và cài đặt Python 3.11.9 thủ công từ: https://www.python.org/downloads/
    echo Lưu ý: Hãy tích chọn ô "Add Python to PATH" khi cài đặt thủ công.
    pause
    exit /b 1
)

:python_ok
echo Sử dụng đường dẫn Python: %PYTHON_EXE%
echo.

:: 2. Check Git installation
set GIT_OK=0
git --version >nul 2>&1
if %errorlevel% equ 0 (
    set GIT_OK=1
    echo Phát hiện Git đã được cài đặt.
) else (
    echo Cảnh báo: Không tìm thấy Git trong hệ thống.
    echo Hệ thống sẽ bỏ qua việc cài đặt các gói ngữ âm tiếng Việt từ GitHub.
    echo Dự án vẫn hoạt động tốt, tính năng XTTSv2 Clone sẽ tự động chuyển sang đọc văn bản gốc.
)
echo.

:: 3. Create virtual environment (.venv)
echo ----------------------------------------------------------------------
echo Đang khởi tạo môi trường ảo (.venv)...
echo ----------------------------------------------------------------------
if not exist .venv (
    %PYTHON_EXE% -m venv .venv
    if %errorlevel% neq 0 (
        echo [LỖI] Khởi tạo môi trường ảo thất bại.
        pause
        exit /b 1
    )
    echo Khởi tạo .venv thành công.
) else (
    echo Thư mục .venv đã tồn tại. Bỏ qua bước khởi tạo.
)
echo.

:: 4. Install libraries
echo ----------------------------------------------------------------------
echo Đang cài đặt các thư viện Python (requirements.txt)...
echo ----------------------------------------------------------------------
.venv\Scripts\python.exe -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [LỖI] Nâng cấp pip thất bại.
)

.venv\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [LỖI] Cài đặt thư viện thất bại.
    pause
    exit /b 1
)
echo Cài đặt các thư viện trong requirements.txt thành công.
echo.

:: 5. Install optional Vietnamese phoneme packages (requires Git)
if !GIT_OK! equ 1 (
    echo ----------------------------------------------------------------------
    echo Đang cài đặt thư viện hỗ trợ phiên âm ngữ âm tiếng Việt (viphoneme)...
    echo ----------------------------------------------------------------------
    .venv\Scripts\python.exe -m pip install git+https://github.com/vunb/viphoneme.git git+https://github.com/vunb/vinorm.git
    if %errorlevel% neq 0 (
        echo [CẢNH BÁO] Cài đặt thư viện phiên âm từ GitHub thất bại.
        echo Hệ thống sẽ tự động sử dụng văn bản gốc khi chạy XTTSv2.
    ) else (
        echo Cài đặt thư viện phiên âm viphoneme thành công.
    )
    echo.
)

:: 6. Download model weights
echo ----------------------------------------------------------------------
echo Đang tiến hành tải trọng số mô hình AI (Piper & XTTSv2)...
echo ----------------------------------------------------------------------
.venv\Scripts\python.exe download_models.py --engine all
if %errorlevel% neq 0 (
    echo [LỖI] Quá trình tải mô hình bị gián đoạn. Bạn có thể tự chạy lệnh:
    echo ".venv\Scripts\python.exe download_models.py --engine all" sau để thử lại.
)
echo.

:: 7. Hardware & GPU Diagnostic
echo ----------------------------------------------------------------------
echo Đang tiến hành chẩn đoán phần cứng (GPU CUDA)...
echo ----------------------------------------------------------------------
.venv\Scripts\python.exe check_gpu.py
echo.

echo ======================================================================
echo THIẾT LẬP THÀNH CÔNG! Dự án AIVoice đã sẵn sàng hoạt động.
echo Để khởi chạy giao diện Wizard tương tác, hãy gõ lệnh:
echo.
echo     .venv\Scripts\python.exe main.py
echo.
echo ======================================================================
pause
