#!/usr/bin/env bash

# Resolve script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SCRIPT_DIR"

# Check parent virtual environment python
if [ -f "../.venv/bin/python" ]; then
    STREAMLIT_CMD="../.venv/bin/python -m streamlit"
else
    STREAMLIT_CMD="python3 -m streamlit"
fi

echo "Starting MediaComposer using parent virtual environment..."
$STREAMLIT_CMD run webui/Main.py --server.port 8502
