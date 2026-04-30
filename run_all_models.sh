#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
fi

# 全部預設模型：baseline 三模型 + 六個 factor ablation models（short/medium/long）
"$PYTHON_BIN" run_pipeline.py \
  --models all \
  --windows all \
  --mode full \
  "$@"
