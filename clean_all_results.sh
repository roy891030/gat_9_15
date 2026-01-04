#!/bin/bash
set -euo pipefail

# 清理常見的訓練產物，避免舊資料干擾新實驗。
# - artifacts_*  : 產生的特徵、模型與 meta
# - experiments/ : 評估與報告輸出
# - plots_*      : 即時視覺化輸出
# - train_*.log  : 訓練日誌
# - RESULTS_SUMMARY.md / results_*.txt : 彙總檔

shopt -s nullglob

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

REMOVE_DIRS=(artifacts_* plots_* experiments)
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
