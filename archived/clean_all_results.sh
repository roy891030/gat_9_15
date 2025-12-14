#!/bin/bash

echo "============================================================"
echo "清理所有舊的訓練結果和 artifacts"
echo "============================================================"

# 停止所有正在運行的訓練進程
echo "1. 停止所有正在運行的訓練..."
pkill -f train_gat_fixed.py || true
pkill -f train_dmfm_wei2022.py || true
sleep 2

# 刪除所有 artifacts 目錄
echo "2. 刪除所有 artifacts 目錄..."
rm -rf gat_artifacts_short
rm -rf gat_artifacts_medium
rm -rf gat_artifacts_long
rm -rf gat_artifacts_full
rm -rf gat_artifacts_out_plus
rm -rf gat_artifacts_wei2022
rm -rf gat_artifacts_out_wei2022
rm -rf gat_artifacts_test

# 刪除所有結果檔案
echo "3. 刪除所有結果檔案..."
rm -f results_*.txt
rm -f train*.log

# 刪除所有圖表目錄
echo "4. 刪除所有圖表目錄..."
rm -rf plots_short_vs_0050
rm -rf plots_medium_vs_0050
rm -rf plots_long_vs_0050
rm -rf plots_gat_vs_0050
rm -rf plots_dmfm
rm -rf plots_attention_wei2022
rm -rf plots_contexts_wei2022
rm -rf plots_attention
rm -rf plots_contexts

# 列出剩餘的重要檔案
echo ""
echo "============================================================"
echo "清理完成！剩餘檔案："
echo "============================================================"
ls -lh *.py *.sh *.md 2>/dev/null | grep -v total

echo ""
echo "✅ 所有舊結果已清理完畢！"
echo "============================================================"
