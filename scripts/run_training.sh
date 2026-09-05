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

"$PYTHON_EXECUTABLE" -m preprocessing.build_dataset \
    --config "$ROOT_DIR/configs/preprocessing.yaml"
"$PYTHON_EXECUTABLE" -m training.team_score.train \
    --config "$ROOT_DIR/configs/team_score.yaml"
"$PYTHON_EXECUTABLE" -m training.minutes_state.train \
    --config "$ROOT_DIR/configs/minutes_state.yaml"
"$PYTHON_EXECUTABLE" -m training.attacking.train \
    --config "$ROOT_DIR/configs/attacking.yaml"
"$PYTHON_EXECUTABLE" -m training.defensive.train \
    --config "$ROOT_DIR/configs/defensive.yaml"
"$PYTHON_EXECUTABLE" -m training.misc_events.train \
    --config "$ROOT_DIR/configs/misc_events.yaml"
"$PYTHON_EXECUTABLE" -m training.bonus.train \
    --config "$ROOT_DIR/configs/bonus.yaml"
