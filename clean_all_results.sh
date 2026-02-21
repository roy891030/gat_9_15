#!/bin/bash
set -euo pipefail

# 清理常見的訓練產物，避免舊資料干擾新實驗。
# - artifacts/   : 新版 pipeline 輸入 artifacts
# - runs/        : 新版 pipeline 輸出（metrics/backtest/plots/logs）
# - artifacts_* / gat_artifacts_* / experiments / plots_* : 舊版輸出
# - train_*.log, results_*.txt, RESULTS_SUMMARY.md : 舊版文字結果

shopt -s nullglob

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

REMOVE_DIRS=(artifacts runs artifacts_* gat_artifacts_* plots_* experiments)
REMOVE_FILES=(train_*.log results_*.txt RESULTS_SUMMARY.md)

for dir in "${REMOVE_DIRS[@]}"; do
  for path in $dir; do
    if [ -d "$path" ]; then
      echo "🧹 移除目錄: $path"
      rm -rf "$path"
    fi
  done
done

for pattern in "${REMOVE_FILES[@]}"; do
  for file in $pattern; do
    if [ -f "$file" ]; then
      echo "🧹 移除檔案: $file"
      rm -f "$file"
    fi
  done
done

echo "✅ 清理完成"
