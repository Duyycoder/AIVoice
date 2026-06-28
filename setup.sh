#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "======================================================================"
echo "         AIVoice Auto-Setup Tool for Unix/macOS (Python 3.11)"
echo "======================================================================"
echo

# 1. Check Python installation
PYTHON_EXE=""
if command -v python3 &>/dev/null; then
    py_ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo "[INFO] Phat hien Python $py_ver."
    if [ "$py_ver" = "3.11" ]; then
        PYTHON_EXE="python3"
    fi
fi

if [ -z "$PYTHON_EXE" ] && command -v python3.11 &>/dev/null; then
    PYTHON_EXE="python3.11"
    echo "[INFO] Phat hien Python 3.11."
fi

if [ -z "$PYTHON_EXE" ]; then
    echo "[ERROR] Khong tim thay Python 3.11 tren may tinh nay."
    echo "AIVoice yeu cau Python 3.11 de dam bao tuong thich thu vien."
    echo "Vui long cai dat Python 3.11 qua trinh quan ly goi cua ban:"
    echo " - macOS: brew install python@3.11"
    echo " - Ubuntu/Debian: sudo apt update && sudo apt install python3.11 python3.11-venv python3.11-dev"
    exit 1
fi

echo "[INFO] Su dung Python: $PYTHON_EXE"
echo

# 2. Check Git installation
GIT_OK=0
if command -v git &>/dev/null; then
    GIT_OK=1
    echo "[INFO] Phat hien Git da duoc cai dat."
else
    echo "[WARNING] Khong tim thay Git. Se bo qua cai dat phien am tu Github."
fi
echo

# 3. Create virtual environment (.venv)
echo "----------------------------------------------------------------------"
echo "[INFO] Dang khoi tao moi truong ao (.venv)..."
echo "----------------------------------------------------------------------"
if [ ! -d ".venv" ]; then
    $PYTHON_EXE -m venv .venv
    echo "[INFO] Khoi tao .venv thanh cong."
else
    echo "[INFO] Thu muc .venv da ton tai."
fi
echo

# Ensure script uses the venv python hereafter
VENV_PYTHON=".venv/bin/python"

# 4. Install pip 24.0 (Required to bypass omegaconf metadata error)
echo "----------------------------------------------------------------------"
echo "[INFO] Dang thiet lap pip phien ban thich hop (pip 24.0)..."
echo "----------------------------------------------------------------------"
$VENV_PYTHON -m pip install --default-timeout=1000 "pip==24.0" || echo "[WARNING] Cai dat pip 24.0 that bai."
echo

# 5. Install libraries
echo "----------------------------------------------------------------------"
echo "[INFO] Dang cai dat cac thu vien Python (requirements.txt)..."
echo "----------------------------------------------------------------------"
$VENV_PYTHON -m pip install --default-timeout=1000 -r requirements.txt
echo "[INFO] Cai dat thu vien thanh cong."
echo

# 5b. Install rvc-python separately with --no-deps
echo "----------------------------------------------------------------------"
echo "[INFO] Dang cai dat rvc-python (--no-deps, tranh xung dot numpy)..."
echo "----------------------------------------------------------------------"
$VENV_PYTHON -m pip install --default-timeout=1000 "rvc-python>=0.1.5" --no-deps || echo "[WARNING] Cai dat rvc-python that bai."
echo

# 6. Install optional Vietnamese phoneme packages (requires Git)
if [ "$GIT_OK" -eq 1 ]; then
    echo "----------------------------------------------------------------------"
    echo "[INFO] Dang cai dat thu vien phien am tieng Viet viphoneme..."
    echo "----------------------------------------------------------------------"
    $VENV_PYTHON -m pip install --default-timeout=1000 git+https://github.com/vunb/viphoneme.git git+https://github.com/vunb/vinorm.git || echo "[WARNING] Cai dat thu vien phien am tu GitHub that bai."
    echo
fi

# 6c. Install Kokoro-Vietnamese (requires Git)
if [ "$GIT_OK" -eq 1 ]; then
    echo "----------------------------------------------------------------------"
    echo "[INFO] Dang thiet lap va cai dat Kokoro-Vietnamese..."
    echo "----------------------------------------------------------------------"
    if [ ! -d "Kokoro-Vietnamese" ]; then
        git clone https://github.com/iamdinhthuan/Kokoro-Vietnamese.git
    fi
    if [ -d "Kokoro-Vietnamese" ]; then
        cd Kokoro-Vietnamese
        ../.venv/bin/pip install --default-timeout=1000 -e .
        cd ..
        echo "[INFO] Cai dat Kokoro-Vietnamese thanh cong."
    else
        echo "[WARNING] Khong the clone Kokoro-Vietnamese tu GitHub."
    fi
    echo
fi

# 6d. Install Valtec-TTS (requires Git)
if [ "$GIT_OK" -eq 1 ]; then
    echo "----------------------------------------------------------------------"
    echo "[INFO] Dang thiet lap va cai dat Valtec-TTS..."
    echo "----------------------------------------------------------------------"
    if [ ! -d "valtec-tts" ]; then
        git clone https://github.com/tronghieuit/valtec-tts.git
    fi
    if [ -d "valtec-tts" ]; then
        cd valtec-tts
        ../.venv/bin/pip install --default-timeout=1000 -e .
        cd ..
        echo "[INFO] Cai dat Valtec-TTS thanh cong."
    else
        echo "[WARNING] Khong the clone Valtec-TTS tu GitHub."
    fi
    echo
fi


# 6b. Copy config.toml from example if not exist
if [ ! -f "MediaComposer/config.toml" ]; then
    echo "----------------------------------------------------------------------"
    echo "[INFO] Dang khoi tao config.toml tu file mau..."
    echo "----------------------------------------------------------------------"
    cp "MediaComposer/config.toml.example" "MediaComposer/config.toml" || echo "[WARNING] Khong the tu dong sao chep config.toml."
    echo "[INFO] Khoi tao config.toml thanh cong! Vui long cap nhat API key trong do neu can."
    echo
fi

# 7. Download model weights
echo "----------------------------------------------------------------------"
echo "[INFO] Dang tai trong so mo hinh AI (Piper & XTTSv2)..."
echo "----------------------------------------------------------------------"
$VENV_PYTHON download_models.py --engine all || echo "[WARNING] Qua trinh tai mo hinh bi gian doan."
echo

# 8. Hardware & GPU Diagnostic
echo "----------------------------------------------------------------------"
echo "[INFO] Dang tien hanh chan doan GPU CUDA..."
echo "----------------------------------------------------------------------"
$VENV_PYTHON check_gpu.py || echo "[WARNING] Chan doan GPU gap loi."
echo

echo "======================================================================"
echo "THIET LAP THANH CONG! AIVoice da san sang hoat dong."
echo "Chay ung dung bang lenh:"
echo
echo "     .venv/bin/python main.py"
echo "======================================================================"
