#!/usr/bin/env bash

# Resolve project directory relative to this script
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJ_DIR"

echo "============================================================"
echo "  AIVoice - Web UI"
echo "  Thu muc: $PROJ_DIR"
echo "============================================================"
echo

# Check virtual environment
if [ ! -f "$PROJ_DIR/.venv/bin/python" ]; then
    echo "[LOI] Khong tim thay .venv/bin/python"
    echo "Vui long chay setup.sh truoc de khoi tao moi truong."
    exit 1
fi

echo "[*] Dang khoi dong server..."
echo "[*] Trinh duyet se tu dong mo khi server san sang."
echo "[*] Dung dong cua so nay khi dang su dung."
echo

"$PROJ_DIR/.venv/bin/python" "$PROJ_DIR/web_ui.py"

echo
echo "[!] Server da dung."
