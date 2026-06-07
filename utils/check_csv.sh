#!/bin/bash

# ============================================================
# CSV 欄位診斷工具
# ============================================================

echo "============================================================"
echo "CSV 欄位診斷"
echo "============================================================"

CSV_PATH="${CSV_PATH:-newData/newData_twse_stock_2014q1_2025q4_appended.csv}"

# 檢查檔案是否存在
if [ ! -f "$CSV_PATH" ]; then
    echo "❌ 找不到 $CSV_PATH"
    echo ""
    echo "請確認："
    echo "1. 檔案是否存在於當前目錄"
    echo "2. 檔案名稱是否正確"
    exit 1
fi

# 檢查是否為 Git LFS 指標檔案
if head -1 "$CSV_PATH" | grep -q "git-lfs"; then
    echo "⚠️  這是 Git LFS 指標檔案，不是實際資料"
    echo ""
    echo "請執行以下步驟："
    echo "1. git lfs pull  # 下載實際檔案"
    echo "或"
    echo "2. 手動上傳 $CSV_PATH 到專案目錄"
    exit 1
fi

# 顯示 CSV 欄位
echo ""
echo "CSV 檔案資訊："
echo "檔案路徑: $CSV_PATH"
echo "檔案大小: $(ls -lh "$CSV_PATH" | awk '{print $5}')"
echo "總行數: $(wc -l < "$CSV_PATH")"
echo ""
echo "前 5 行："
head -5 "$CSV_PATH"

echo ""
echo "============================================================"
echo "欄位名稱："
echo "============================================================"
head -1 "$CSV_PATH" | tr ',' '\n' | nl

echo ""
echo "============================================================"
echo "建議："
echo "============================================================"
echo ""
echo "請將上述欄位名稱提供給我，我會幫你修改 build_artifacts.py"
echo ""
echo "或者，你可以手動修改 build_artifacts.py 中的 COLMAPS："
echo "  位置: build_artifacts.py:83-118"
echo "============================================================"
