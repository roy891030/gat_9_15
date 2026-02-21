#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ -x "$SCRIPT_DIR/.venv/bin/python" ]; then
  PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
fi

# 與 run_all_models.sh 同流程；保留檔名供舊習慣使用。
"$PYTHON_BIN" run_pipeline.py \
  --models all \
  --baseline_models linear,xgboost,lstm \
  --windows all \
  --mode full \
  "$@"
