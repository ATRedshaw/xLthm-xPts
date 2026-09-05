#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/inference.yaml}"

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

MODEL_ARTIFACTS=(
    "$ROOT_DIR/data/models/team_score/model.joblib"
    "$ROOT_DIR/data/models/minutes_state/model.joblib"
    "$ROOT_DIR/data/models/attacking/model.joblib"
    "$ROOT_DIR/data/models/defensive/model.joblib"
    "$ROOT_DIR/data/models/misc_events/model.joblib"
    "$ROOT_DIR/data/models/bonus/model.joblib"
)

models_exist=true
for artifact in "${MODEL_ARTIFACTS[@]}"; do
    if [[ ! -f "$artifact" ]]; then
        models_exist=false
        break
    fi
done

if [[ "$models_exist" != true ]]; then
    "$ROOT_DIR/scripts/run_training.sh"
fi

"$PYTHON_EXECUTABLE" -m inference.build_features --config "$CONFIG_PATH"
"$PYTHON_EXECUTABLE" -m inference.predict_components --config "$CONFIG_PATH"
"$PYTHON_EXECUTABLE" -m inference.generate_xpts --config "$CONFIG_PATH"
