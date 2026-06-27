#!/usr/bin/env bash

# Resolve project directory relative to this script
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJ_DIR"

echo "============================================================"
echo "  AIVoice - Running Integration Tests Suite"
echo "  Project Directory: $PROJ_DIR"
echo "============================================================"
echo

# Check virtual environment
if [ ! -f "$PROJ_DIR/.venv/bin/python" ]; then
    echo "[ERROR] Virtual environment not found at: $PROJ_DIR/.venv"
    echo "Vui long chay setup.sh truoc."
    exit 1
fi

"$PROJ_DIR/.venv/bin/python" tests/run_tests.py
