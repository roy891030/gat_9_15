#!/bin/bash

# ============================================================
# DMFM (Wei et al. 2022) 完整執行流程
# ============================================================

set -e  # 遇到錯誤立即停止

echo "============================================================"
echo "DMFM (Wei et al. 2022) 完整實驗流程"
echo "============================================================"

# 設定參數
DEVICE="cuda"
INDUSTRY_CSV="unique_2019q3to2025q3.csv"
PRICES_CSV="unique_2019q3to2025q3.csv"
ARTIFACT_DIR="gat_artifacts_wei2022"
START_DATE="2019-09-16"
END_DATE="2025-09-12"
HORIZON=5

# 訓練參數
EPOCHS=200
LR=1e-3
HIDDEN_DIM=64
HEADS=2
DROPOUT=0.1
LAMBDA_ATTN=0.1
LAMBDA_IC=1.0

echo ""
echo "實驗參數:"
echo "  資料期間: $START_DATE ~ $END_DATE"
echo "  預測視野: $HORIZON 日"
echo "  裝置: $DEVICE"
echo "  訓練週期: $EPOCHS"
echo "  學習率: $LR"
echo "============================================================"

# ============================================================
# Step 1: 建立 Artifacts（使用新的預處理方式）
# ============================================================
echo ""
echo "Step 1: 建立 Artifacts（不做截面標準化，保留原始特徵）"
echo "------------------------------------------------------------"

python build_artifacts.py \
  --prices $PRICES_CSV \
  --industry_csv $INDUSTRY_CSV \
  --artifact_dir $ARTIFACT_DIR \
  --start_date $START_DATE \
  --end_date $END_DATE \
  --horizon $HORIZON

if [ $? -eq 0 ]; then
    echo "✓ Step 1 完成"
else
    echo "✗ Step 1 失敗"
    exit 1
fi

# ============================================================
# Step 2: 訓練 DMFM 模型
# ============================================================
echo ""
echo "Step 2: 訓練 DMFM (Wei et al. 2022) 模型"
echo "------------------------------------------------------------"

python train_dmfm_wei2022.py \
  --artifact_dir $ARTIFACT_DIR \
  --epochs $EPOCHS \
  --lr $LR \
  --device $DEVICE \
  --hidden_dim $HIDDEN_DIM \
  --heads $HEADS \
  --dropout $DROPOUT \
  --lambda_attn $LAMBDA_ATTN \
  --lambda_ic $LAMBDA_IC

if [ $? -eq 0 ]; then
    echo "✓ Step 2 完成"
else
    echo "✗ Step 2 失敗"
    exit 1
fi

# ============================================================
# Step 3: 視覺化 Factor Attention
# ============================================================
echo ""
echo "Step 3: 視覺化 Factor Attention 權重"
echo "------------------------------------------------------------"

python visualize_factor_attention.py \
  --artifact_dir $ARTIFACT_DIR \
  --output_dir plots_attention_wei2022 \
  --device cpu \
  --top_k 15

if [ $? -eq 0 ]; then
    echo "✓ Step 3 完成"
else
    echo "✗ Step 3 失敗"
    exit 1
fi

# ============================================================
# Step 4: 分析階層式特徵
# ============================================================
echo ""
echo "Step 4: 分析階層式特徵 (C, C_I, C_U)"
echo "------------------------------------------------------------"

python analyze_contexts.py \
  --artifact_dir $ARTIFACT_DIR \
  --output_dir plots_contexts_wei2022 \
  --device cpu \
  --sample_days 20

if [ $? -eq 0 ]; then
    echo "✓ Step 4 完成"
else
    echo "✗ Step 4 失敗"
    exit 1
fi

# ============================================================
# Step 5: 評估模型指標（使用現有的 evaluate_metrics.py）
# ============================================================
echo ""
echo "Step 5: 評估模型指標"
echo "------------------------------------------------------------"

# 檢查 evaluate_metrics.py 是否支援 DMFM_Wei2022
# 如果不支援，則跳過此步驟

if [ -f "evaluate_metrics.py" ]; then
    echo "使用 evaluate_metrics.py 評估模型..."
    python evaluate_metrics.py \
      --artifact_dir $ARTIFACT_DIR \
      --weights $ARTIFACT_DIR/dmfm_wei2022_best.pt \
      --device $DEVICE \
      --industry_csv $INDUSTRY_CSV \
      > results_dmfm_wei2022_metrics.txt || echo "⚠ evaluate_metrics.py 可能不支援 DMFM_Wei2022，跳過"
else
    echo "⚠ 找不到 evaluate_metrics.py，跳過此步驟"
fi

# ============================================================
# Step 6: 投資組合回測（使用現有的 evaluate_portfolio.py）
# ============================================================
echo ""
echo "Step 6: 投資組合回測"
echo "------------------------------------------------------------"

if [ -f "evaluate_portfolio.py" ]; then
    echo "使用 evaluate_portfolio.py 進行回測..."
    python evaluate_portfolio.py \
      --artifact_dir $ARTIFACT_DIR \
      --weights $ARTIFACT_DIR/dmfm_wei2022_best.pt \
      --device $DEVICE \
      --top_pct 0.10 \
      --rebalance_days 5 \
      --industry_csv $INDUSTRY_CSV \
      > results_dmfm_wei2022_portfolio.txt || echo "⚠ evaluate_portfolio.py 可能不支援 DMFM_Wei2022，跳過"
else
    echo "⚠ 找不到 evaluate_portfolio.py，跳過此步驟"
fi

# ============================================================
# 完成
# ============================================================
echo ""
echo "============================================================"
echo "所有步驟完成！"
echo "============================================================"

echo ""
echo "生成的檔案："
echo "  模型檔案："
echo "    - $ARTIFACT_DIR/dmfm_wei2022_best.pt"
echo "    - $ARTIFACT_DIR/train_log_wei2022.txt"
echo ""
echo "  視覺化圖表："
echo "    - plots_attention_wei2022/"
echo "    - plots_contexts_wei2022/"
echo ""
echo "  評估結果："
if [ -f "results_dmfm_wei2022_metrics.txt" ]; then
    echo "    - results_dmfm_wei2022_metrics.txt"
fi
if [ -f "results_dmfm_wei2022_portfolio.txt" ]; then
    echo "    - results_dmfm_wei2022_portfolio.txt"
fi

echo ""
echo "查看 Factor Attention 分析："
echo "  cat plots_attention_wei2022/factor_attention_summary.txt"
echo ""
echo "查看階層式特徵分析："
echo "  cat plots_contexts_wei2022/context_analysis_summary.txt"
echo ""
echo "============================================================"
