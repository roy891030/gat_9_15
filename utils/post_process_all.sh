#!/bin/bash

# ============================================================
# 訓練完成後的視覺化和評估
# ============================================================

echo "============================================================"
echo "後處理：視覺化和評估所有模型"
echo "============================================================"
echo "開始時間: $(date)"
echo "============================================================"

# ============================================================
# 短期 DMFM
# ============================================================
if [ -f "gat_artifacts_short/dmfm_wei2022_best.pt" ]; then
    echo ""
    echo "====== 處理短期 DMFM ======"

    echo "[1/2] 視覺化 Factor Attention..."
    python visualize_factor_attention.py \
      --artifact_dir gat_artifacts_short \
      --output_dir plots_short_attention \
      --device cpu \
      --top_k 15

    echo "[2/2] 分析階層式特徵..."
    python analyze_contexts.py \
      --artifact_dir gat_artifacts_short \
      --output_dir plots_short_contexts \
      --device cpu \
      --sample_days 20

    echo "✅ 短期 DMFM 處理完成"
else
    echo "⚠ 找不到短期 DMFM 模型，跳過"
fi

# ============================================================
# 中期 DMFM
# ============================================================
if [ -f "gat_artifacts_medium/dmfm_wei2022_best.pt" ]; then
    echo ""
    echo "====== 處理中期 DMFM ======"

    echo "[1/2] 視覺化 Factor Attention..."
    python visualize_factor_attention.py \
      --artifact_dir gat_artifacts_medium \
      --output_dir plots_medium_attention \
      --device cpu \
      --top_k 15

    echo "[2/2] 分析階層式特徵..."
    python analyze_contexts.py \
      --artifact_dir gat_artifacts_medium \
      --output_dir plots_medium_contexts \
      --device cpu \
      --sample_days 20

    echo "✅ 中期 DMFM 處理完成"
else
    echo "⚠ 找不到中期 DMFM 模型，跳過"
fi

# ============================================================
# 長期 DMFM
# ============================================================
if [ -f "gat_artifacts_long/dmfm_wei2022_best.pt" ]; then
    echo ""
    echo "====== 處理長期 DMFM ======"

    echo "[1/2] 視覺化 Factor Attention..."
    python visualize_factor_attention.py \
      --artifact_dir gat_artifacts_long \
      --output_dir plots_long_attention \
      --device cpu \
      --top_k 15

    echo "[2/2] 分析階層式特徵..."
    python analyze_contexts.py \
      --artifact_dir gat_artifacts_long \
      --output_dir plots_long_contexts \
      --device cpu \
      --sample_days 20

    echo "✅ 長期 DMFM 處理完成"
else
    echo "⚠ 找不到長期 DMFM 模型，跳過"
fi

# ============================================================
# GATRegressor（如果存在）
# ============================================================
if [ -f "gat_artifacts_gat/gat_regressor.pt" ]; then
    echo ""
    echo "====== 處理 GATRegressor ======"

    if [ -f "evaluate_metrics.py" ]; then
        echo "[1/2] 評估指標..."
        python evaluate_metrics.py \
          --artifact_dir gat_artifacts_gat \
          --weights gat_artifacts_gat/gat_regressor.pt \
          --device cuda \
          --industry_csv unique_2019q3to2025q3.csv \
          > results_gat_metrics.txt 2>&1 || echo "⚠ 評估失敗"
    fi

    if [ -f "evaluate_portfolio.py" ]; then
        echo "[2/2] 投資組合回測..."
        python evaluate_portfolio.py \
          --artifact_dir gat_artifacts_gat \
          --weights gat_artifacts_gat/gat_regressor.pt \
          --device cuda \
          --top_pct 0.10 \
          --rebalance_days 5 \
          --industry_csv unique_2019q3to2025q3.csv \
          > results_gat_portfolio.txt 2>&1 || echo "⚠ 回測失敗"
    fi

    echo "✅ GATRegressor 處理完成"
else
    echo "⚠ 找不到 GATRegressor 模型，跳過"
fi

# ============================================================
# 生成總結報告
# ============================================================
echo ""
echo "====== 生成總結報告 ======"

cat > RESULTS_SUMMARY.md <<'EOF'
# 訓練結果總結

## 📊 模型訓練完成狀態

| 模型 | 資料期間 | 狀態 | 訓練日誌 | 視覺化 |
|------|---------|------|---------|--------|
| DMFM (短期) | 2019-2020 | ✓ | train_short.log | plots_short_* |
| DMFM (中期) | 2019-2022 | ✓ | train_medium.log | plots_medium_* |
| DMFM (長期) | 2019-2025 | ✓ | train_long.log | plots_long_* |
| GATRegressor | 2019-2022 | ✓ | train_gat.log | - |

## 📈 Factor Attention 分析

### 短期 DMFM
EOF

if [ -f "plots_short_attention/factor_attention_summary.txt" ]; then
    echo '```' >> RESULTS_SUMMARY.md
    head -20 plots_short_attention/factor_attention_summary.txt >> RESULTS_SUMMARY.md
    echo '```' >> RESULTS_SUMMARY.md
else
    echo "尚未生成" >> RESULTS_SUMMARY.md
fi

cat >> RESULTS_SUMMARY.md <<'EOF'

### 中期 DMFM
EOF

if [ -f "plots_medium_attention/factor_attention_summary.txt" ]; then
    echo '```' >> RESULTS_SUMMARY.md
    head -20 plots_medium_attention/factor_attention_summary.txt >> RESULTS_SUMMARY.md
    echo '```' >> RESULTS_SUMMARY.md
else
    echo "尚未生成" >> RESULTS_SUMMARY.md
fi

cat >> RESULTS_SUMMARY.md <<'EOF'

### 長期 DMFM
EOF

if [ -f "plots_long_attention/factor_attention_summary.txt" ]; then
    echo '```' >> RESULTS_SUMMARY.md
    head -20 plots_long_attention/factor_attention_summary.txt >> RESULTS_SUMMARY.md
    echo '```' >> RESULTS_SUMMARY.md
else
    echo "尚未生成" >> RESULTS_SUMMARY.md
fi

cat >> RESULTS_SUMMARY.md <<'EOF'

## 🔬 階層式特徵分析

### 變異數降低效果

| 模型 | 原始 (C) | 產業中性 (C_I) | 全市場中性 (C_U) | 總降低 |
|------|---------|---------------|----------------|--------|
| 短期 | - | - | - | - |
| 中期 | - | - | - | - |
| 長期 | - | - | - | - |

詳細請查看：plots_*_contexts/context_analysis_summary.txt

## 📁 檔案結構

```
生成的檔案：
├── gat_artifacts_short/        # 短期模型
├── gat_artifacts_medium/       # 中期模型
├── gat_artifacts_long/         # 長期模型
├── gat_artifacts_gat/          # GATRegressor
├── plots_short_attention/      # 短期 Factor Attention
├── plots_medium_attention/     # 中期 Factor Attention
├── plots_long_attention/       # 長期 Factor Attention
├── plots_short_contexts/       # 短期階層式特徵分析
├── plots_medium_contexts/      # 中期階層式特徵分析
├── plots_long_contexts/        # 長期階層式特徵分析
├── train_short.log             # 短期訓練日誌
├── train_medium.log            # 中期訓練日誌
├── train_long.log              # 長期訓練日誌
└── train_gat.log               # GATRegressor 訓練日誌
```

## 🎯 下一步

1. 查看訓練日誌確認收斂：
   ```bash
   tail -50 train_*.log
   ```

2. 查看 Factor Attention 分析：
   ```bash
   cat plots_*_attention/factor_attention_summary.txt
   ```

3. 查看階層式特徵分析：
   ```bash
   cat plots_*_contexts/context_analysis_summary.txt
   ```

4. 比較不同資料期間的結果

5. 選擇最佳模型進行投資組合回測
EOF

echo "✅ 總結報告已生成：RESULTS_SUMMARY.md"

# ============================================================
# 完成
# ============================================================
echo ""
echo "============================================================"
echo "後處理完成！"
echo "結束時間: $(date)"
echo "============================================================"

echo ""
echo "📊 生成的檔案："
echo ""
ls -d plots_* 2>/dev/null && echo "" || echo "  (無圖表)"
ls -lh *.md 2>/dev/null && echo "" || echo ""
ls -lh results_*.txt 2>/dev/null && echo "" || echo "  (無評估結果)"

echo ""
echo "============================================================"
echo "查看總結報告："
echo "  cat RESULTS_SUMMARY.md"
echo "============================================================"
