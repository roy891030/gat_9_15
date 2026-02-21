#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
fi

# 核心版：只跑 GAT + DMFM（short/medium/long）
"$PYTHON_BIN" run_pipeline.py \
  --models gat,dmfm \
  --windows all \
  --mode full \
  "$@"
