#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_EXECUTABLE="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/venv/bin/python" ]]; then
    PYTHON_EXECUTABLE="$ROOT_DIR/venv/bin/python"
elif [[ -x "$ROOT_DIR/venv/Scripts/python.exe" ]]; then
    PYTHON_EXECUTABLE="$ROOT_DIR/venv/Scripts/python.exe"
else
    PYTHON_EXECUTABLE="python"
fi

export PYTHONPATH="$ROOT_DIR/backend/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT_DIR"

"$PYTHON_EXECUTABLE" -m api
